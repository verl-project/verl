"""Stream and independently verify complete saved evaluation outputs on CPU."""

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

import pyarrow.parquet as pq


def verify(run):
    sys.path.insert(0, str(run/'source'))
    from examples.search_r1.eval_reporting import DATASET_COUNTS, MULTIHOP, check_final_provenance, select_eval_shard
    from examples.search_r1.protocol import score_final_turn

    manifest = json.loads((run/'manifest.json').read_text())
    final = json.loads((run/'evaluation.json').read_text())
    assert (run/'exit-code.txt').read_text().strip() == '0'
    merged = json.loads((run/'merged-checkpoint-verification.json').read_text())
    assert merged['verified'] and merged['checkpoint_step'] == final['checkpoint_step'] == 200
    assert merged['checkpoint'] == manifest['checkpoint']
    assert merged['merged_model'] == final['model_path'] == manifest['model']
    assert merged['tokenizer_vocab_and_template_equal']
    hashes_checked = 0
    for field, base in [('source_sha256',run/'source'),('core_file_sha256',run/'source'),
                         ('auxiliary_sha256',run)]:
        for name, expected in manifest[field].items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest() == expected, name
            hashes_checked += 1
    data = Path(manifest['eval_data'])
    preflight = json.loads((run/'data-preflight.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == preflight['file_sha256']
    rows = pq.read_table(data).to_pylist()
    assert Counter(r['data_source'] for r in rows) == Counter(DATASET_COUNTS)
    for index, row in enumerate(rows):
        row['eval_id'] = index
    rows = select_eval_shard(rows, manifest.get('eval_shard_index', 0), manifest.get('eval_num_shards', 1))
    assert len(rows) == manifest.get('eval_shard_expected_count', len(rows))
    assert final.get('eval_shard_index', 0) == manifest.get('eval_shard_index', 0)
    assert final.get('eval_num_shards', 1) == manifest.get('eval_num_shards', 1)
    by_id = {row['eval_id']: row for row in rows}
    expected_counts = Counter(r['data_source'] for r in rows)
    counts, correct, searches, invalid = Counter(), Counter(), Counter(), Counter()
    seen = set()
    paths = sorted((run/'results').glob('worker-*.jsonl'))
    assert {p.name for p in paths} == {f'worker-{i}.jsonl' for i in range(8)}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                result = json.loads(line)
                index = result['eval_id']
                assert isinstance(index,int) and index in by_id and index not in seen
                original = by_id[index]
                source = original['data_source']
                assert result['data_source'] == source
                assert result['input_messages'] == original['prompt']
                assert result['ground_truth'] == original['reward_model']['ground_truth']
                assert not result.get('error')
                check_final_provenance(result['messages'], original['prompt'],
                                       result['final_assistant'], result['termination'])
                score = score_final_turn(result['final_assistant'], original['reward_model']['ground_truth'],
                                         result['termination'])
                assert result['em'] == score['score']
                assert result['answer_valid'] == score['answer_valid']
                counts[source] += 1
                correct[source] += score['score']
                searches[source] += result['search_calls']
                invalid[source] += not result['answer_valid']
                seen.add(index)
    assert seen == set(by_id)
    assert counts == expected_counts
    for name,count in counts.items():
        reported = final['per_dataset'][name]
        assert reported['count'] == count
        assert reported['correct'] == correct[name]
        assert reported['em'] == correct[name]/count
        assert reported['mean_search_calls'] == searches[name]/count
        assert reported['invalid_answer_count'] == invalid[name]
    derived = {'overall_macro_em': statistics.mean(correct[n]/counts[n] for n in counts),
               'overall_micro_em': sum(correct.values())/len(rows),
               'multihop_macro_em': statistics.mean(correct[n]/counts[n] for n in MULTIHOP),
               'multihop_micro_em': sum(correct[n] for n in MULTIHOP)/sum(counts[n] for n in MULTIHOP)}
    for key, expected in derived.items():
        assert final[key] == expected, key
    assert final['total_count'] == len(rows)
    assert final['total_correct'] == sum(correct.values())
    assert final['multihop_count'] == sum(counts[n] for n in MULTIHOP)
    report = {'verified':True, 'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'count':len(seen),'counts':dict(counts),'correct':dict(correct),
              'source_and_auxiliary_hashes_checked':hashes_checked,'eval_data_hash_verified':True,
              'checks':['no missing or duplicate questions','original source, prompt and accepted answers',
                        'last sampled assistant provenance','normalized final-answer EM recomputed',
                        'macro and micro averages recomputed','checkpoint step200 and tokenizer verification',
                        'frozen recipe, core code, auxiliary and data hashes unchanged','worker exit code zero'],
              **derived}
    (run/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run.resolve())

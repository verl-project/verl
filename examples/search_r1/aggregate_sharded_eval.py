"""Recompute final-answer EM across all ten complete, independently verified shards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

import pyarrow.parquet as pq


def main(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    sys.path.insert(0,manifest['source_directory'])
    from examples.search_r1.eval_reporting import DATASET_COUNTS, MULTIHOP, check_final_provenance
    from examples.search_r1.protocol import score_final_turn
    data=Path(manifest['eval_data'])
    assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['eval_data_sha256']
    rows=pq.read_table(data).to_pylist()
    assert Counter(r['data_source'] for r in rows)==Counter(DATASET_COUNTS)
    seen=set();counts=Counter();correct=Counter();searches=Counter();invalid=Counter()
    reports=[]
    for shard in manifest['shards']:
        run=Path(shard['run_dir']);local=json.loads((run/'manifest.json').read_text())
        assert (run/'exit-code.txt').read_text().strip()=='0',run
        verified=json.loads((run/'independent-verification.json').read_text());assert verified['verified']
        report=json.loads((run/'evaluation.json').read_text());assert report['verified']
        assert report['eval_num_shards']==10 and report['eval_shard_index']==shard['index']
        assert report['total_count']==shard['expected_count']
        assert local['checkpoint']==manifest['checkpoint'] and local['model']==manifest['model']
        before=len(seen)
        paths=sorted((run/'results').glob('worker-*.jsonl'));assert len(paths)==8
        for path in paths:
            for line in path.open():
                result=json.loads(line);index=result['eval_id']
                assert isinstance(index,int) and 0<=index<len(rows) and index not in seen
                assert index%10==shard['index'],(index,shard['index'])
                original=rows[index];source=original['data_source']
                assert result['data_source']==source and result['input_messages']==original['prompt']
                assert result['ground_truth']==original['reward_model']['ground_truth']
                assert not result.get('error')
                check_final_provenance(result['messages'],original['prompt'],result['final_assistant'],result['termination'])
                score=score_final_turn(result['final_assistant'],result['ground_truth'],result['termination'])
                assert score['score']==result['em'] and score['answer_valid']==result['answer_valid']
                seen.add(index);counts[source]+=1;correct[source]+=score['score'];searches[source]+=result['search_calls'];invalid[source]+=not result['answer_valid']
        assert len(seen)-before==shard['expected_count']
        reports.append({'index':shard['index'],'questions':report['total_count'],'elapsed_s':report['elapsed_s']})
    assert seen==set(range(51713)) and counts==Counter(DATASET_COUNTS)
    per={name:{'count':counts[name],'correct':int(correct[name]),'em':correct[name]/counts[name],
        'em_percent':100*correct[name]/counts[name],'mean_search_calls':searches[name]/counts[name],
        'invalid_answer_count':invalid[name]} for name in DATASET_COUNTS}
    multi=sum(counts[name] for name in MULTIHOP)
    final={'verified':True,'checkpoint_step':200,'checkpoint':manifest['checkpoint'],'model_path':manifest['model'],
        'total_count':len(seen),'total_correct':int(sum(correct.values())),'per_dataset':per,
        'overall_macro_em':statistics.mean(v['em'] for v in per.values()),'overall_micro_em':sum(correct.values())/len(seen),
        'multihop_macro_em':statistics.mean(per[name]['em'] for name in MULTIHOP),
        'multihop_micro_em':sum(correct[name] for name in MULTIHOP)/multi,'multihop_count':multi,
        'shards':reports,'scoring':'Final-assistant normalized EM, recomputed across original global IDs; no shard-score averaging.'}
    (batch/'evaluation.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path)
    main(parser.parse_args().batch.resolve())

"""Exercise the final merger on complete synthetic seven-dataset output."""
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.search_r1.eval_reporting import DATASET_COUNTS


def test_full_merge_recomputes_em_and_rejects_duplicate_global_id(tmp_path,capsys):
    script=Path(__file__).resolve().parents[1]/'aggregate_sharded_eval.py'
    spec=importlib.util.spec_from_file_location('sharded_eval_merger_test',script)
    merger=importlib.util.module_from_spec(spec);spec.loader.exec_module(merger)
    rows=[]
    for source,count in DATASET_COUNTS.items():
        for _ in range(count):
            i=len(rows)
            rows.append({'data_source':source,'prompt':[{'role':'user','content':f'Question {i}'}],
                         'reward_model':{'ground_truth':{'target':['Paris']}}})
    data=tmp_path/'data.parquet';pq.write_table(pa.Table.from_pylist(rows),data)
    shards=[]
    for index in range(10):
        run=tmp_path/f'shard-{index}';(run/'results').mkdir(parents=True)
        ids=list(range(index,len(rows),10));shards.append({'index':index,'run_dir':str(run),'expected_count':len(ids)})
        (run/'exit-code.txt').write_text('0')
        (run/'manifest.json').write_text(json.dumps({'checkpoint':'c','model':'m'}))
        (run/'independent-verification.json').write_text(json.dumps({'verified':True}))
        (run/'evaluation.json').write_text(json.dumps({'verified':True,'eval_num_shards':10,'eval_shard_index':index,'total_count':len(ids),'elapsed_s':1}))
        with ExitStack() as stack:
            files=[stack.enter_context((run/'results'/f'worker-{i}.jsonl').open('w')) for i in range(8)]
            for local,i in enumerate(ids):
                correct=i%2==0;final='<answer>Paris</answer>' if correct else '<answer>Rome</answer>'
                result={'eval_id':i,'data_source':rows[i]['data_source'],'input_messages':rows[i]['prompt'],
                        'ground_truth':rows[i]['reward_model']['ground_truth'],
                        'messages':rows[i]['prompt']+[{'role':'assistant','content':final}],
                        'final_assistant':final,'termination':'answer','em':float(correct),'answer_valid':1.,'search_calls':0}
                files[local%8].write(json.dumps(result)+'\n')
    import hashlib
    manifest={'source_directory':str(Path(__file__).resolve().parents[3]),'eval_data':str(data),
              'eval_data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'checkpoint':'c','model':'m','shards':shards}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    # Verify one non-contiguous shard with the production per-job verifier too.
    # Its largest global IDs exceed the local shard length, catching renumbering.
    from collections import Counter
    from examples.search_r1.eval_reporting import aggregate_results
    run=tmp_path/'shard-7'
    selected=[{**rows[i], 'eval_id':i} for i in range(7,len(rows),10)]
    report=aggregate_results(selected,sorted((run/'results').glob('worker-*.jsonl')),Counter(r['data_source'] for r in selected))
    report.update(checkpoint_step=200,model_path='m',eval_shard_index=7,eval_num_shards=10,elapsed_s=1)
    (run/'evaluation.json').write_text(json.dumps(report))
    local={'checkpoint':'c','model':'m','eval_data':str(data),'eval_shard_index':7,'eval_num_shards':10,
           'eval_shard_expected_count':len(selected),'source_sha256':{},'core_file_sha256':{},'auxiliary_sha256':{}}
    (run/'manifest.json').write_text(json.dumps(local))
    (run/'data-preflight.json').write_text(json.dumps({'file_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}))
    (run/'merged-checkpoint-verification.json').write_text(json.dumps({'verified':True,'checkpoint_step':200,
        'checkpoint':'c','merged_model':'m','tokenizer_vocab_and_template_equal':True}))
    verifier_path=script.parent/'verify_evaluation.py'
    verifier_spec=importlib.util.spec_from_file_location('shard_verifier_test',verifier_path)
    verifier=importlib.util.module_from_spec(verifier_spec);verifier_spec.loader.exec_module(verifier)
    verifier.verify(run)
    merger.main(tmp_path)
    final=json.loads((tmp_path/'evaluation.json').read_text())
    assert final['total_count']==51713 and final['total_correct']==25857
    assert final['overall_micro_em']==25857/51713
    assert final['multihop_count']==22523
    capsys.readouterr()
    first=tmp_path/'shard-0/results/worker-0.jsonl'
    with first.open() as stream:duplicate=stream.readline()
    with first.open('a') as stream:stream.write(duplicate)
    with pytest.raises(AssertionError):merger.main(tmp_path)

import asyncio
import copy
import json

import pytest

from examples.search_r1.eval_reporting import aggregate_results, check_final_provenance, result_record, run_bounded
from examples.search_r1.tests.test_search_r1 import FakeServer, FakeTool, make_loop, run_inputs
from examples.search_r1.tests.test_search_r1 import tokenizer as tokenizer


def test_bounded_evaluation_preserves_ids_and_cancels_siblings_on_error():
    async def check(fail):
        active=0;peak=0;seen=[]
        async def item(index):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            try:
                await asyncio.sleep(.001)
                if fail and index==3:raise ValueError('generation failed')
                seen.append(index)
            finally:active-=1
        if fail:
            with pytest.raises(ValueError,match='generation failed'):
                await run_bounded(list(range(25)),item,4)
        else:
            await run_bounded(list(range(25)),item,4)
            assert sorted(seen)==list(range(25))
        assert peak==4 and active==0
    asyncio.run(check(False));asyncio.run(check(True))


def fixture_results():
    rows=[];results=[]
    for i,(source,correct) in enumerate([('nq',True),('hotpotqa',False),('bamboogle',True)]):
        prompt=[{'role':'user','content':f'Question {i}'}]
        final='<answer>Paris</answer>' if correct else '<answer>Rome</answer>'
        rows.append({'eval_id':i,'data_source':source,'prompt':prompt,'reward_model':{'ground_truth':{'target':['Paris']}}})
        results.append({'eval_id':i,'data_source':source,'input_messages':prompt,'ground_truth':{'target':['Paris']},
                        'messages':prompt+[{'role':'tool','content':'Paris is the gold answer.'},{'role':'assistant','content':final}],
                        'final_assistant':final,'termination':'answer','em':float(correct),'answer_valid':1,
                        'search_calls':1})
    return rows,results


@pytest.mark.parametrize('corruption',[None,'duplicate','missing','wrong_score','wrong_gold','wrong_prompt'])
def test_aggregation_recomputes_final_answer_em_and_requires_complete_coverage(tmp_path,corruption):
    rows,results=fixture_results()
    if corruption=='duplicate':results.append(copy.deepcopy(results[0]))
    if corruption=='missing':results.pop()
    if corruption=='wrong_score':results[1]['em']=1
    if corruption=='wrong_gold':results[0]['ground_truth']={'target':['Rome']}
    if corruption=='wrong_prompt':results[0]['input_messages']=[{'role':'user','content':'Other'}]
    path=tmp_path/'results.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in reversed(results)))
    expected={'nq':1,'hotpotqa':1,'bamboogle':1}
    if corruption:
        with pytest.raises(AssertionError):aggregate_results(rows,[path],expected)
    else:
        report=aggregate_results(rows,[path],expected)
        assert report['total_correct']==2 and report['per_dataset']['hotpotqa']['em']==0
        assert report['overall_macro_em']==pytest.approx(2/3)
        assert report['overall_micro_em']==pytest.approx(2/3)
        assert report['multihop_macro_em']==.5 and report['multihop_micro_em']==.5


def test_budget_or_empty_termination_can_end_without_final_assistant():
    prompt=[{'role':'user','content':'Question'}]
    check_final_provenance(prompt,prompt,'','empty')
    assistant={'role':'assistant','content':'<search>evidence</search>'}
    check_final_provenance(prompt+[assistant,{'role':'tool','content':'Paris'}],prompt,
                           assistant['content'],'context_budget')
    with pytest.raises(AssertionError):
        check_final_provenance(prompt,prompt,'<answer>Paris</answer>','answer')


def test_native_search_agent_output_is_scored_without_postprocessing(request):
    loaded_tokenizer=request.getfixturevalue('tokenizer')
    async def check():
        server=FakeServer(loaded_tokenizer,['<search>capital of France</search>','<answer>Rome</answer>'])
        loop=make_loop(loaded_tokenizer,server,FakeTool())
        inputs=run_inputs()
        output=await loop.run({'temperature':0,'logprobs':False},**inputs)
        row={'eval_id':0,'data_source':'nq','prompt':inputs['raw_prompt'],'reward_model':inputs['reward_model']}
        result=result_record(row,output,.1)
        assert result['em']==0 and result['search_calls']==1
        assert result['messages'][-1]['content']=='<answer>Rome</answer>'
        assert 'Paris' in result['messages'][-2]['content']
    asyncio.run(check())

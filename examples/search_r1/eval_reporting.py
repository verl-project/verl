"""Streaming evaluation scheduling and independent final-answer EM verification."""

import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from .protocol import score_final_turn

DATASET_COUNTS = {'nq': 3610, 'triviaqa': 11313, 'popqa': 14267, 'hotpotqa': 7405,
                  '2wikimultihopqa': 12576, 'musique': 2417, 'bamboogle': 125}
MULTIHOP = ['hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']


def check_final_provenance(messages, prompt, final, termination):
    assert messages[:len(prompt)] == prompt
    assistants = [m for m in messages[len(prompt):] if m['role'] == 'assistant']
    assert final == (assistants[-1]['content'] if assistants else '')
    if termination == 'answer':
        assert messages[-1]['role'] == 'assistant'


async def run_bounded(items, function, limit):
    if limit < 1:
        raise ValueError('Concurrency must be positive')
    iterator = iter(items)

    async def consume():
        for item in iterator:
            await function(item)

    tasks = [asyncio.create_task(consume()) for _ in range(min(limit, len(items)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def result_record(row, output, elapsed):
    extra = output.extra_fields
    messages = extra['search_r1_messages']
    final = extra['search_r1_final_assistant']
    check_final_provenance(messages, row['prompt'], final, extra['search_r1_termination'])
    score = score_final_turn(final, row['reward_model']['ground_truth'], extra['search_r1_termination'])
    assert score['score'] == output.reward_score
    return {'eval_id': row['eval_id'], 'data_source': row['data_source'],
            'input_messages': row['prompt'], 'messages': messages,
            'ground_truth': row['reward_model']['ground_truth'], 'final_assistant': final,
            'termination': extra['search_r1_termination'], 'em': score['score'],
            'answer_valid': score['answer_valid'],
            'search_calls': extra['reward_extra_info']['search_calls'],
            'assistant_turns': extra['reward_extra_info']['assistant_turns'],
            'generated_tokens': sum(output.response_mask), 'wall_s': elapsed,
            'metrics': output.metrics.model_dump(), 'tool_metrics': extra['search_r1_tool_metrics']}


def aggregate_results(rows, paths, expected_counts=None):
    expected_counts = expected_counts or DATASET_COUNTS
    assert Counter(r['data_source'] for r in rows) == Counter(expected_counts)
    by_id = {r['eval_id']: r for r in rows}
    assert len(by_id) == len(rows)
    seen, groups = set(), defaultdict(list)
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                result = json.loads(line)
                index = result['eval_id']
                assert index in by_id and index not in seen, f'Unexpected or duplicate eval_id {index}'
                assert not result.get('error'), result
                original = by_id[index]
                assert result['data_source'] == original['data_source']
                assert result['input_messages'] == original['prompt']
                assert result['ground_truth'] == original['reward_model']['ground_truth']
                check_final_provenance(result['messages'], original['prompt'], result['final_assistant'], result['termination'])
                score = score_final_turn(result['final_assistant'], original['reward_model']['ground_truth'], result['termination'])
                assert result['em'] == score['score'], 'Saved EM disagrees with independent recomputation'
                seen.add(index)
                groups[result['data_source']].append(result)
    assert seen == set(by_id), f'Missing {len(set(by_id)-seen)} evaluation results'
    per_dataset = {}
    for name, count in expected_counts.items():
        values = groups[name]
        assert len(values) == count
        correct = sum(r['em'] for r in values)
        per_dataset[name] = {'count': count, 'correct': int(correct), 'em': correct/count,
                             'em_percent': 100*correct/count,
                             'mean_search_calls': statistics.mean(r['search_calls'] for r in values),
                             'invalid_answer_count': sum(not r['answer_valid'] for r in values)}
    total_correct = sum(v['correct'] for v in per_dataset.values())
    multi = [per_dataset[name] for name in MULTIHOP if name in per_dataset]
    multi_count = sum(v['count'] for v in multi)
    return {'verified': True, 'total_count': len(rows), 'total_correct': total_correct,
            'per_dataset': per_dataset,
            'overall_macro_em': statistics.mean(v['em'] for v in per_dataset.values()),
            'overall_micro_em': total_correct/len(rows),
            'multihop_datasets': [n for n in MULTIHOP if n in per_dataset],
            'multihop_count': multi_count,
            'multihop_macro_em': statistics.mean(v['em'] for v in multi) if multi else None,
            'multihop_micro_em': sum(v['correct'] for v in multi)/multi_count if multi else None,
            'scoring': 'Normalized exact match on the final assistant <answer>; observations are never scored.'}

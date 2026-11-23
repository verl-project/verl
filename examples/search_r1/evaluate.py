"""Evaluate the unchanged SearchR1AgentLoop using standalone verl vLLM replicas."""

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap, ToolListWrap
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.workers.rollout.llm_server import LLMServerManager

from .agent_loop import SearchR1AgentLoop
from .eval_reporting import DATASET_COUNTS, aggregate_results, result_record, run_bounded


@ray.remote(num_cpus=1)
class EvaluationWorker(AgentLoopWorker):
    def __init__(self, config, client, worker_id):
        super().__init__(config, client)
        self.worker_id = worker_id

    async def evaluate(self, rows, limit, directory):
        directory = Path(directory)
        completed = Counter()
        started = time.perf_counter()
        loop_settings = OmegaConf.load(self.rollout_config.agent.agent_loop_config_path)[0]
        settings = {key: value for key, value in loop_settings.items() if key not in ['name', '_target_']}
        params = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'logprobs': False, 'repetition_penalty': 1.0}
        path = directory/f'worker-{self.worker_id}.jsonl'
        assert not path.exists(), 'Refusing to overwrite evaluation results'
        with path.open('w', buffering=1) as stream:
            async def evaluate_one(row):
                tick = time.perf_counter()
                loop = SearchR1AgentLoop(
                    trainer_config=DictConfigWrap(self.config), server_manager=self.llm_client,
                    tokenizer=self.tokenizer, processor=self.processor, hf_model_type=self.hf_model_type,
                    dataset_cls=self.dataset_cls, data_config=DictConfigWrap(self.config.data),
                    tools=ToolListWrap(self.tools), **settings)
                output = await loop.run(params, raw_prompt=row['prompt'], reward_model=row['reward_model'],
                                        extra_info=row['extra_info'])
                record = result_record(row, output, time.perf_counter()-tick)
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                completed[row['data_source']] += 1
                total = sum(completed.values())
                if total % 100 == 0:
                    progress = {'worker': self.worker_id, 'completed': total, 'total': len(rows),
                                'per_dataset': dict(completed), 'elapsed_s': time.perf_counter()-started}
                    (directory/f'worker-{self.worker_id}-progress.json').write_text(json.dumps(progress)+'\n')
                    print('EVAL_PROGRESS', json.dumps(progress), flush=True)
            try:
                await run_bounded(rows, evaluate_one, limit)
            finally:
                for tool in self.tools:
                    close = getattr(tool, 'close', None)
                    if close:
                        await close()
        return {'worker': self.worker_id, 'completed': sum(completed.values()), 'file': str(path)}


async def evaluate(config):
    run = Path(os.environ['RUN_DIR'])
    rows = pq.read_table(os.environ['EVAL_DATA']).to_pylist()
    assert Counter(r['data_source'] for r in rows) == Counter(DATASET_COUNTS)
    for i, row in enumerate(rows):
        row['eval_id'] = i
    started = time.perf_counter()
    manager = await LLMServerManager.create(config)
    print('EVAL_SERVERS_READY', manager.get_addresses(), flush=True)
    count = int(os.environ.get('EVAL_WORKERS', '8'))
    per_worker = int(os.environ.get('EVAL_CONCURRENCY_PER_WORKER', '1024'))
    directory = run/'results';directory.mkdir(exist_ok=True)
    workers = [EvaluationWorker.options(name=f'search_r1_eval_worker_{i}').remote(config, manager.get_client(), i)
               for i in range(count)]
    results = await asyncio.gather(*[worker.evaluate.remote(rows[i::count], per_worker, str(directory))
                                     for i, worker in enumerate(workers)])
    report = aggregate_results(rows, [r['file'] for r in results])
    report.update(checkpoint_step=200, model_path=config.actor_rollout_ref.model.path,
                  elapsed_s=time.perf_counter()-started, worker_results=results,
                  global_trajectory_concurrency=count*per_worker,
                  max_num_seqs_per_engine=config.actor_rollout_ref.rollout.max_num_seqs,
                  gpu_memory_utilization=config.actor_rollout_ref.rollout.gpu_memory_utilization)
    (run/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('EVALUATION_COMPLETE', json.dumps(report), flush=True)


@hydra.main(config_path='../../verl/trainer/config', config_name='ppo_trainer', version_base=None)
def main(config):
    OmegaConf.resolve(config)
    (Path(os.environ['RUN_DIR'])/'resolved-eval-config.json').write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2)+'\n')
    ray.init(num_cpus=60, runtime_env=get_ppo_ray_runtime_env(config))
    try:
        asyncio.run(evaluate(config))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()

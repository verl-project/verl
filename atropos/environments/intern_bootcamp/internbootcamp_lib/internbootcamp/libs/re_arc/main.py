import time
import tqdm
import os
import json

from random import seed as set_seed

from . import dsl
from .dsl import *

from . import utils
from .utils import *

from . import generators
from . import verifiers



def get_generators() -> dict:
    """
    returns mapper from task identifiers (keys) to example generator functions
    """
    prefix = 'generate_'
    return {
        strip_prefix(n, prefix): getattr(generators, n) for n in dir(generators) if n.startswith(prefix)
    }


def get_verifiers() -> dict:
    """
    returns mapper from task identifiers (keys) to example verifier functions
    """
    prefix = 'verify_'
    return {
        strip_prefix(n, prefix): getattr(verifiers, n) for n in dir(verifiers) if n.startswith(prefix)
    }


def get_rng_difficulty(
    example: dict
) -> float:
    """
    RNG-Difficulty: proxy measure for example difficulty, defined as the mean of sampled floats within example generation
    """
    rng = getattr(utils, 'rng')
    setattr(utils, 'rng', [])
    return sum(rng) / len(rng)


def get_pso_difficulty(
    example: dict
) -> float:
    """
    PSO-Difficulty: proxy measure for example difficulty, defined as weighted sum of #Pixels, #Symbols, #Objects
    """
    i, o = example['input'], example['output']
    hwi = height(i) * width(i)
    hwo = height(o) * width(o)
    pix_pct = (hwi + hwo) / 1800
    col_pct = len(palette(i) | palette(o)) / 10
    obj_dens = (len(objects(i, T, F, F)) / hwi + len(objects(o, T, F, F)) / hwo) / 2
    return (pix_pct + col_pct + obj_dens) / 3


def demo_generator(key, n=6):
    with open(f'arc_original/training/{key}.json', 'r') as fp:
        original_task = json.load(fp)
    original_task = original_task['train'] + original_task['test']
    generator = getattr(generators, f'generate_{key}')
    generated_examples = [generator(0, 1) for k in range(n)]
    plot_task(original_task)
    plot_task(generated_examples)
    

def generate_dataset(
    path: str = 're_arc',
    seed: int = 42,
    n_examples: int = 1000,
    diff_lb: float = 0,
    diff_ub: float = 1
) -> None:
    """
    generates dataset

    path: which folder to save data to
    seed: for deterministic generation / reproducibility
    n_examples: number of examples per task
    diff_lb: lower bound for difficulty
    diff_ub: upper bound for difficulty
    """
    set_seed(seed)
    os.makedirs(path)
    tasks_path = os.path.join(path, 'tasks')
    os.makedirs(tasks_path)
    generators_mapper = get_generators()
    verifiers_mapper = get_verifiers()
    keys = sorted(generators_mapper.keys())
    k = len(keys)
    desc = f'task 0/{k}, example 0/{n_examples}'
    pbar = tqdm.tqdm(enumerate(keys), desc=desc, position=0, leave=True, total=k)
    metadata = dict()
    for i, key in pbar:
        generator = generators_mapper[key]
        verifier = verifiers_mapper[key]
        seen = set()
        examples = []
        stats = {
            'n_generations': 0, 'n_verified': 0, 'n_nondegenerate': 0,
            'rng_difficulties': [], 'pso_difficulties': []
        }
        start = time.time()
        while len(examples) < n_examples:
            example, identifier, success = None, None, True
            try:
                example = generator(diff_lb, diff_ub)
                assert is_grid(example['input'])
                assert is_grid(example['output'])
                identifier = hash(example['input'])
                stats['n_generations'] += 1
            except:
                success = False
            try:
                assert success and verifier(example['input']) == example['output']
                stats['n_verified'] += 1
            except:
                success = False
            try:
                assert success and example['input'] != example['output']
                stats['n_nondegenerate'] += 1
            except:
                success = False
            if success and identifier not in seen:
                examples.append(example)
                seen.add(identifier)
                stats['rng_difficulties'].append(get_rng_difficulty(example))
                stats['pso_difficulties'].append(get_pso_difficulty(example))
                desc = f'task {i+1}/{k}, example {len(examples)}/{n_examples}'
                pbar.set_description(desc)
        end = time.time()
        stats['runtime'] = end - start
        with open(os.path.join(tasks_path, f'{key}.json'), 'w') as fp:
            json.dump(examples, fp)
        metadata[key] = stats
    with open(os.path.join(path, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)


def demo_dataset(
    folder: str = 're_arc',
    n: int = 8,
    s: int = 0,
    e: int = 400
) -> None:
    """
    visualizing snippets from a generated dataset (original, easy, medium and hard instances for each task)
    """
    with open(f'{folder}/metadata.json', 'r') as fp:
        metadata = json.load(fp)
    for i, fn in enumerate(sorted(os.listdir(f'{folder}/tasks'))):
        if s <= i < e:
            key = fn[:8]
            with open(f'arc_original/training/{key}.json', 'r') as fp:
                original_task = json.load(fp)
            with open(f'{folder}/tasks/{key}.json', 'r') as fp:
                generated_task = json.load(fp)
            original_task = [format_example(example) for example in original_task['train'] + original_task['test']]
            generated_task = [format_example(example) for example in generated_task[:10*n]]
            difficulties = metadata[key]['pso_difficulties'][:9*n]
            generated_task = [ex for ex, diff in sorted(zip(generated_task, difficulties), key=lambda item: item[1])]
            easy = generated_task[1*n:2*n]
            hard = generated_task[8*n:9*n]
            print(key)
            print('original:')
            plot_task(original_task)
            print('generated (easy):')
            plot_task(easy)
            print('generated (hard):')
            plot_task(hard)


def evaluate_verifiers_on_original_tasks() -> None:
    """
    runs the verifiers on the original ARC training tasks
    """
    verifiers = get_verifiers()
    dataset = dict()
    for key in verifiers.keys():
        with open(f'arc_original/training/{key}.json', 'r') as fp:
            task = json.load(fp)
        dataset[key] = format_task(task)
    fix_bugs(dataset)
    failed_on = set()
    for key, verifier in verifiers.items():
        task = dataset[key]
        try:
            for example in task['train'] + task['test']:
                assert verifier(example['input']) == example['output']
        except:
            failed_on.add(key)
    n = len(dataset)
    k = len(failed_on)
    print(f'verification programs work for all examples for {n-k}/{n} tasks')
    print(f'verification fails (on one example) for tasks {failed_on}')


import json
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterator, Protocol


AnswerFormatter = Callable[[str, str], str]


def prompt_with_answer_format(question: str, answer_format: str) -> str:
    """Render the canonical BlindTasks text prompt."""
    return f"{question} {answer_format}"


def answer_prompt(
    question: str,
    answer_description: str,
    example: str,
    fmt: AnswerFormatter,
) -> str:
    return prompt_with_answer_format(question, fmt(answer_description, example))


def count_prompt(question: str, fmt: AnswerFormatter) -> str:
    return answer_prompt(question, "a single integer", "6", fmt)


@dataclass(frozen=True)
class RenderConfig:
    """Per-instance visual parameters used by every renderer"""
    image_size: int = 256
    line_width: int = 2
    bg: tuple[int, int, int] = (255, 255, 255)
    fg: tuple[int, int, int] = (0, 0, 0)
    accent: tuple[int, int, int] = (220, 0, 0)


@dataclass(frozen=True)
class TaskInstance:
    """One instance of a blind-task: (image, prompt, ground truth)"""
    name: str
    seed: int
    question: str
    ground_truth: dict
    image: Image.Image | None = None
    image_path: Path | None = None

    def __post_init__(self) -> None:
        if self.image is None and self.image_path is None:
            raise ValueError("TaskInstance requires at least one between image and image_path")


class TaskType(Protocol):
    """Implemented by every blind-task"""
    name: str

    def sample(self, seed: int) -> TaskInstance:
        """Return a reproducible instance"""
        ...

    def prompt(self, instance: TaskInstance, fmt: AnswerFormatter) -> str:
        """Return a prompt whose answer schema is formatted by fmt."""
        ...
    
    def verify(self, instance: TaskInstance, prediction: str) -> float:
        """Return a reward in [0.0, 1.0]"""
        ...

def stream(task: TaskType, *, start: int = 0) -> Iterator[TaskInstance]:
    """Indefinitely yield instances starting from seed=start"""
    seed = start
    while True:
        yield task.sample(seed)
        seed += 1

def dump_instances(instances: list["TaskInstance"], task_name: str, *, output_dir: Path) -> None:
    """Write a pre-sampled list of instances to output_dir"""
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for instance in instances:
        question_id = f"{task_name}_s{instance.seed}"
        image = instance.image
        if image is None:
            image = Image.open(instance.image_path)
        image.convert("RGB").save(image_dir / f"{question_id}.jpg", "JPEG")

        records.append({
            "question_id": question_id,
            "question": instance.question,
            "answers": [str(instance.ground_truth["answer"])],
            "image_file": f"images/{question_id}.jpg",
            "extra": {
                "task_name": task_name,
                "seed": instance.seed,
                "ground_truth": instance.ground_truth,
            },
        })

    with open(output_dir / "metadata.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(records)} samples to {output_dir}")


def dump(task: TaskType, n: int, *, output_dir: Path, start_seed: int = 0) -> None:
    """Write n samples (sequential seed) to output_dir as metadata.jsonl + images/"""
    instances = [task.sample(seed) for seed in range(start_seed, start_seed + n)]
    dump_instances(instances, task.name, output_dir=output_dir)

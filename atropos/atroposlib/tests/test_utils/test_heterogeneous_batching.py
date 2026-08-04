import random

from atroposlib.api.utils import grab_exact_from_heterogeneous_queue


def test_grab_exact_from_heterogeneous_queue():
    "randomly samples from the space of potential inputs to grab_exact_from_heterogeneous_queue"
    for random_bs in range(10000):
        bs = 64 * random.randint(1, 20)
        queue = []
        for i in range(random.randint(1, 100)):
            # queue.append(
            #     {
            #         "tokens": [[2 * i] for _ in range(2)],
            #     }
            # )
            queue.append(
                {
                    "tokens": [[2 * i + 1] for _ in range(8)],
                }
            )
        batch, queue = grab_exact_from_heterogeneous_queue(queue, bs)
        if random_bs == 0:
            print(batch)
        if batch is not None:
            assert (
                sum(len(item["tokens"]) for item in batch) == bs
            ), f"expected batch size {bs}, got {len(batch)}"

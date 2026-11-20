import pytest
import requests

from testing.api.utils import launch_api_for_testing


def register_data(group="test", proj="test", batch_size=32) -> requests.Response:
    x = requests.post(
        "http://localhost:8000/register",
        json={"wandb_group": group, "wandb_project": proj, "batch_size": batch_size},
    )
    return x


def post_scored_data(
    tokens=((0,),), masks=((0,),), scores=(0,), ref_logprobs=((0,),)
) -> requests.Response:
    data = {
        "tokens": tokens,
        "masks": masks,
        "scores": scores,
    }
    if ref_logprobs is not None:
        data["ref_logprobs"] = ref_logprobs
    x = requests.post("http://localhost:8000/scored_data", json=data)
    return x


def reset() -> requests.Response:
    x = requests.get("http://localhost:8000/reset_data")
    return x


@pytest.fixture(scope="session")
def api():
    proc = launch_api_for_testing()
    yield
    proc.kill()


def test_register(api):
    x = register_data()
    assert x.status_code == 200, x.text
    data = x.json()
    assert "uuid" in data


def test_reset(api):
    x = register_data()
    assert x.status_code == 200, x.text
    data = x.json()
    assert "uuid" in data
    x = post_scored_data()
    assert x.status_code == 200, x.text
    x = reset()
    print("0-0-0-0-0-0-0-0", flush=True)
    print(x.text, flush=True)
    print("0-0-0-0-0-0-0-0", flush=True)
    assert x.status_code == 200, x.text
    x = requests.get("http://localhost:8000/info")
    assert x.status_code == 200
    assert x.json()["batch_size"] == -1
    x = requests.get("http://localhost:8000/status")
    assert x.status_code == 200, x.text
    data = x.json()
    assert data["current_step"] == 0
    assert data["queue_size"] == 0
    x = requests.get("http://localhost:8000/wandb_info")
    assert x.status_code == 200, x.text
    data = x.json()
    assert data["group"] is None
    assert data["project"] is None


def test_batch_size(api):
    x = register_data()
    assert x.status_code == 200, x.text
    # get the batch size
    x = requests.get("http://localhost:8000/info")

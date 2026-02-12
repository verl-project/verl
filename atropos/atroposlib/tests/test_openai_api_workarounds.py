import asyncio
import os

import dotenv
import pytest

from atroposlib.envs.server_handling.openai_server import APIServerConfig, OpenAIServer


@pytest.mark.providers
def test_openai_api_n_kwarg_ignore_discovery():
    dotenv.load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    config = APIServerConfig(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model_name="openai/gpt-4.1-nano",
        timeout=1200,
        num_max_requests_at_once=512,
        num_requests_for_eval=64,
        rolling_buffer_length=1024,
    )
    assert not config.n_kwarg_is_ignored, "n kwarg is not ignored by default"
    n = 4
    server = OpenAIServer(
        config=config,
    )
    response = asyncio.run(
        server.chat_completion(
            messages=[
                {"role": "user", "content": "Hello, how are you?"},
            ],
            n=n,
        )
    )
    assert server.config.n_kwarg_is_ignored, "n kwarg is should be set after discovery"
    print(len(response.choices), n)
    assert (
        len(response.choices) == n
    ), f"Expected {n} responses, got {len(response.choices)}"


@pytest.mark.providers
def test_openai_api_n_kwarg_ignore_use():
    dotenv.load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    config = APIServerConfig(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model_name="openai/gpt-4.1-nano",
        timeout=1200,
        num_max_requests_at_once=512,
        num_requests_for_eval=64,
        rolling_buffer_length=1024,
        n_kwarg_is_ignored=True,
    )
    server = OpenAIServer(
        config=config,
    )
    n = 4
    response = asyncio.run(
        server.chat_completion(
            messages=[
                {"role": "user", "content": "Hello, how are you?"},
            ],
            n=n,
        )
    )
    assert server.config.n_kwarg_is_ignored, "n kwarg is should be set after discovery"
    assert (
        len(response.choices) == n
    ), f"Expected {n} responses, got {len(response.choices)}"


@pytest.mark.providers
def test_openai_api_n_kwarg_supported():
    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")
    config = APIServerConfig(
        model_name="gpt-4.1-nano",
        timeout=1200,
        num_max_requests_at_once=512,
        num_requests_for_eval=64,
        rolling_buffer_length=1024,
        n_kwarg_is_ignored=False,
    )
    server = OpenAIServer(
        config=config,
    )
    n = 4
    response = asyncio.run(
        server.chat_completion(
            messages=[
                {"role": "user", "content": "Hello, how are you?"},
            ],
            n=n,
        )
    )
    assert (
        not server.config.n_kwarg_is_ignored
    ), "n kwarg should be used with supported models"
    assert (
        len(response.choices) == n
    ), f"Expected {n} responses, got {len(response.choices)}"

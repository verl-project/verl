import asyncio
import warnings

import aiohttp
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic_cli import FailedExecutionException

from atroposlib.envs.constants import NAMESPACE_SEP, OPENAI_NAMESPACE
from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig


class OpenAIServer(APIServer):
    """
    OpenAI server handling.
    """

    def __init__(self, config: APIServerConfig):
        self.openai = openai.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        super().__init__(config)

    async def check_server_status_task(self, chat_completion: bool = True):
        while True:
            try:
                if chat_completion:
                    await self.openai.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                    )
                else:
                    await self.openai.completions.create(
                        model=self.config.model_name,
                        prompt="hi",
                        max_tokens=1,
                    )
                self.server_healthy = True
            except (
                aiohttp.ClientError,
                openai.OpenAIError,
                openai.APITimeoutError,
                Exception,
            ):
                self.server_healthy = False
            await asyncio.sleep(1)

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the chat completion using the openai client.
        """
        assert (
            kwargs.get("model", None) is not None
        ), "Model is required for chat completion!"
        assert (
            kwargs.get("messages", None) is not None
        ), "Messages are required for chat completion!"
        if self.config.n_kwarg_is_ignored:
            n = kwargs.pop("n", 1)
            completion_list = await asyncio.gather(
                *[self.openai.chat.completions.create(**kwargs) for _ in range(n)]
            )
            completions = completion_list[0]
            if n > 1:
                for c in completion_list[1:]:
                    completions.choices.extend(c.choices)
            else:
                completions = await self.openai.chat.completions.create(**kwargs)
        else:
            if "n" in kwargs:
                n = kwargs["n"]
            else:
                n = 1
            completions = await self.openai.chat.completions.create(**kwargs)
            if len(completions.choices) != n:
                if len(completions.choices) != 1:
                    raise ValueError(
                        f"Expected 1 or {n} completions, got {len(completions.choices)}!"
                    )
                else:
                    warnings.warn("n kwarg is ignored by the API, setting to True")
                    self.config.n_kwarg_is_ignored = True
                    completion_list = await asyncio.gather(
                        *[
                            self.openai.chat.completions.create(**kwargs)
                            for _ in range(1, n)
                        ]
                    )
                    for c in completion_list:
                        completions.choices.extend(c.choices)
        return completions

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Wrapper for the completion using the openai client.
        """
        assert (
            kwargs.get("model", None) is not None
        ), "Model is required for completion!"
        assert (
            kwargs.get("prompt", None) is not None
        ), "Prompt is required for completion!"
        if self.config.n_kwarg_is_ignored:
            n = kwargs.pop("n", 1)
            completion_list = await asyncio.gather(
                *[self.openai.completions.create(**kwargs) for _ in range(n)]
            )
            completions = completion_list[0]
            if n > 1:
                for c in completion_list[1:]:
                    completions.choices.extend(c.choices)
        else:
            if "n" in kwargs:
                n = kwargs["n"]
            else:
                n = 1
            completions = await self.openai.completions.create(**kwargs)
            if len(completions.choices) != n:
                if len(completions.choices) != 1:
                    raise ValueError(
                        f"Expected 1 or {n} completions, got {len(completions.choices)}!"
                    )
                else:
                    warnings.warn("n kwarg is ignored by the API, setting to True")
                    self.config.n_kwarg_is_ignored = True
                    completion_list = await asyncio.gather(
                        *[self.openai.completions.create(**kwargs) for _ in range(1, n)]
                    )
                    for c in completion_list:
                        completions.choices.extend(c.choices)
        return completions


def resolve_openai_configs(
    default_server_configs,
    openai_config_dict,
    yaml_config,
    cli_passed_flags,
    logger,
):
    """
    Helper to resolve the final server_configs, handling single, multiple servers, and overrides.
    """
    from atroposlib.envs.server_handling.server_manager import ServerBaseline

    openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"
    openai_yaml_config = yaml_config.get(OPENAI_NAMESPACE, None)
    openai_cli_config = {
        k: v for k, v in cli_passed_flags.items() if k.startswith(openai_full_prefix)
    }

    is_multi_server_yaml = (
        isinstance(openai_yaml_config, list) and len(openai_yaml_config) >= 2
    )
    is_multi_server_default = (
        (not is_multi_server_yaml)
        and isinstance(default_server_configs, list)
        and len(default_server_configs) >= 2
    )

    if (is_multi_server_yaml or is_multi_server_default) and openai_cli_config:
        raise FailedExecutionException(
            message=f"CLI overrides for OpenAI settings (--{openai_full_prefix}*) are not supported "
            f"when multiple servers are defined (either via YAML list under '{OPENAI_NAMESPACE}' "
            "or a default list with length >= 2).",
            exit_code=2,
        )

    if is_multi_server_yaml:
        logger.info(
            f"Using multi-server configuration defined in YAML under '{OPENAI_NAMESPACE}'."
        )
        try:
            server_configs = [APIServerConfig(**cfg) for cfg in openai_yaml_config]
        except Exception as e:
            raise FailedExecutionException(
                f"Error parsing multi-server OpenAI configuration from YAML under '{OPENAI_NAMESPACE}': {e}"
            ) from e
    elif isinstance(default_server_configs, ServerBaseline):
        logger.info("Using ServerBaseline configuration.")
        server_configs = default_server_configs
    elif is_multi_server_default:
        logger.info("Using default multi-server configuration (length >= 2).")
        server_configs = default_server_configs
    else:
        logger.info(
            "Using single OpenAI server configuration based on merged settings (default/YAML/CLI)."
        )
        try:
            final_openai_config = APIServerConfig(**openai_config_dict)
        except Exception as e:
            raise FailedExecutionException(
                f"Error creating final OpenAI configuration from merged settings: {e}\n"
                f"Merged Dict: {openai_config_dict}"
            ) from e

        if isinstance(default_server_configs, APIServerConfig):
            server_configs = final_openai_config
        elif isinstance(default_server_configs, list):
            server_configs = [final_openai_config]
        else:
            logger.warning(
                f"Unexpected type for default_server_configs: {type(default_server_configs)}. "
                f"Proceeding with single OpenAI server configuration based on merged settings."
            )
            server_configs = [final_openai_config]

    return server_configs

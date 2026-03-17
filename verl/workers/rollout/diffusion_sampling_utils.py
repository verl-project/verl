from typing import Any


def build_diffusion_backend_sampling_params(
    sampling_params: dict[str, Any],
    *,
    model_extra_configs: dict[str, Any] | None,
    direct_param_names: set[str],
    rename_map: dict[str, str],
) -> dict[str, Any]:
    """Translate generic diffusion request params into backend sampling kwargs.

    Generic request fields stay in the agent loop. Backend/model-specific fields are
    attached here, where request-level overrides should win over model defaults.
    """
    backend_sampling_params: dict[str, Any] = {}
    extra_args = {
        key: value for key, value in (model_extra_configs or {}).items() if value is not None
    }

    for key, value in sampling_params.items():
        if value is None:
            continue

        backend_key = rename_map.get(key, key)
        if backend_key in direct_param_names:
            backend_sampling_params[backend_key] = value
        else:
            extra_args[backend_key] = value

    if extra_args:
        backend_sampling_params["extra_args"] = extra_args

    return backend_sampling_params

from __future__ import annotations

from typing import Callable

from .draft_server_patch import (
    init_tokenizer_manager as draft_init_tokenizer_manager,
    run_detokenizer_process as draft_run_detokenizer_process,
    run_scheduler_process as draft_run_scheduler_process,
)
from .verify_server_patch import (
    init_tokenizer_manager as verify_init_tokenizer_manager,
    run_detokenizer_process as verify_run_detokenizer_process,
    run_scheduler_process as verify_run_scheduler_process,
)


def get_sglang_launch_components(server_role: str | None) -> dict[str, Callable]:
    if server_role == "verify":
        return {
            "init_tokenizer_manager_func": verify_init_tokenizer_manager,
            "run_scheduler_process_func": verify_run_scheduler_process,
            "run_detokenizer_process_func": verify_run_detokenizer_process,
        }
    if server_role == "draft":
        return {
            "init_tokenizer_manager_func": draft_init_tokenizer_manager,
            "run_scheduler_process_func": draft_run_scheduler_process,
            "run_detokenizer_process_func": draft_run_detokenizer_process,
        }
    return {}

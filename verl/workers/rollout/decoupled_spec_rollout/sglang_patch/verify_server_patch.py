def init_tokenizer_manager(*args, **kwargs):
    from sglang.srt.entrypoints.engine import init_tokenizer_manager as upstream

    return upstream(*args, **kwargs)


def run_scheduler_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_scheduler_process as upstream

    return upstream(*args, **kwargs)


def run_detokenizer_process(*args, **kwargs):
    from sglang.srt.entrypoints.engine import run_detokenizer_process as upstream

    return upstream(*args, **kwargs)

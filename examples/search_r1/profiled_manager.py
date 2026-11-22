"""Use the standard TQ manager, with driver phase markers for this experiment."""

import os

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

from .profiling import install_phase_timers


class ProfiledAgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_phase_timers(os.environ["RUN_DIR"])

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only ALFWorld environment service; install in its own Python environment.

Run: ALFWORLD_DATA=/data/alfworld python examples/group_delivery/alfworld_server.py
The default bind address is loopback. Use a single server worker because sessions are local.
"""

import hashlib
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Synthetic tail: real agentic workloads mix fast tasks with a small fraction
# of very slow ones; ALFWorld's own spread is only ~1.3x (P95/P50), which never
# exercises group_delivery's overlap. These knobs inject a deterministic
# per-task tool latency (same task -> same delay in every run and every mode),
# changing wall-clock time only, never the environment dynamics or rewards.
TAIL_FRACTION = float(os.environ.get("ALFWORLD_TAIL_FRACTION", "0") or 0)
TAIL_DELAY_MS = int(os.environ.get("ALFWORLD_TAIL_DELAY_MS", "0") or 0)
# A client that cancels right after POST /sessions never reaches the DELETE in
# its finally block, so the session would occupy the 128-session cap forever.
# Sessions idle longer than this are closed and evicted on the next creation.
SESSION_TTL_S = float(os.environ.get("ALFWORLD_SESSION_TTL_S", "1800") or 0)


def tail_delay_s(game_file: str) -> float:
    """Deterministic slow-task delay from the game file hash (0 for fast tasks)."""
    if TAIL_FRACTION <= 0 or TAIL_DELAY_MS <= 0:
        return 0.0
    digest = int(hashlib.sha1(game_file.encode()).hexdigest(), 16)
    if (digest % 1000) >= TAIL_FRACTION * 1000:
        return 0.0
    # +-20% jitter so the tail is a distribution, not a single spike.
    jitter = 0.8 + 0.4 * ((digest >> 20) % 1000) / 1000.0
    return TAIL_DELAY_MS * jitter / 1000.0


def make_environment(game_file, max_steps):
    """Construct the official text environment without privileged expert observations."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(
        [str(game_file)],
        infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)


class OpenSession(BaseModel):
    """Path relative to ALFWORLD_DATA, supplied by the prepared dataset."""

    game_file: str


class Action(BaseModel):
    """One environment action; model text never controls the reward."""

    action: str


def create_app(data_root, max_steps=50, environment_factory=make_environment):
    """Create isolated sessions; an injectable factory allows CPU lifecycle tests."""
    root = Path(data_root).resolve()
    sessions, registry_lock = {}, threading.RLock()
    # TextWorld's shared TatSu parser is not thread-safe; concurrent env.step()
    # calls corrupt its grammar stack. FastAPI runs sync endpoints in a thread
    # pool, so all environment access must be serialized process-wide.
    env_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        with env_lock:
            for session in sessions.values():
                session["env"].close()

    app = FastAPI(lifespan=lifespan)

    def view(session):
        return {key: value for key, value in session.items() if key not in ("env", "lock", "last_used")}

    def purge_expired_sessions():
        """Evict sessions whose client disappeared without sending DELETE.

        Each candidate is re-checked under its session lock, which a step or a
        GET holds for the whole call, so an in-flight session is never evicted
        even when one call outlives the TTL. Eviction closes the environment
        under the process-wide env lock.
        """
        if SESSION_TTL_S <= 0:
            return
        now = time.time()
        with registry_lock:
            candidates = [
                (session_id, session)
                for session_id, session in sessions.items()
                if now - session["last_used"] > SESSION_TTL_S
            ]
        for session_id, session in candidates:
            if not session["lock"].acquire(blocking=False):
                continue
            try:
                if time.time() - session["last_used"] <= SESSION_TTL_S:
                    continue
                with registry_lock:
                    if sessions.get(session_id) is not session:
                        continue
                    sessions.pop(session_id, None)
                with env_lock:
                    try:
                        session["env"].close()
                    except Exception:
                        pass
            finally:
                session["lock"].release()

    def lookup(session_id):
        with registry_lock:
            if session_id not in sessions:
                raise HTTPException(404, "Unknown environment session")
            return sessions[session_id]

    @app.post("/sessions")
    def open_session(body: OpenSession):
        path = (root / body.game_file).resolve()
        if not path.is_relative_to(root) or path.name != "game.tw-pddl" or not path.is_file():
            raise HTTPException(400, "Expected a game.tw-pddl file inside ALFWORLD_DATA")
        # Sleep before taking registry_lock/env_lock: a slow task must not
        # serialize session creation for every other concurrent episode.
        delay_s = tail_delay_s(body.game_file)
        if delay_s:
            time.sleep(delay_s)
        purge_expired_sessions()
        with registry_lock:
            if len(sessions) >= 128:
                raise HTTPException(503, "Too many active environment sessions")
            env = environment_factory(path, max_steps)
            try:
                with env_lock:
                    observations, infos = env.reset()
            except Exception:
                env.close()
                raise
            session_id = uuid4().hex
            sessions[session_id] = dict(
                env=env,
                lock=threading.RLock(),
                session_id=session_id,
                observation=observations[0],
                commands=list(infos["admissible_commands"][0]),
                steps=0,
                max_steps=max_steps,
                won=False,
                done=False,
                error=False,
                last_used=time.time(),
            )
            return view(sessions[session_id])

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = lookup(session_id)
        with session["lock"]:
            with registry_lock:
                if sessions.get(session_id) is not session:
                    raise HTTPException(404, "Unknown environment session")
            session["last_used"] = time.time()
            return view(session)

    @app.post("/sessions/{session_id}/step")
    def step_session(session_id: str, body: Action):
        session = lookup(session_id)
        with session["lock"]:
            # lookup() released registry_lock before this lock was taken, so a
            # concurrent open_session could have evicted the session already.
            with registry_lock:
                if sessions.get(session_id) is not session:
                    raise HTTPException(404, "Unknown environment session")
            session["last_used"] = time.time()
            try:
                if not session["done"]:
                    try:
                        with env_lock:
                            observations, _, dones, infos = session["env"].step([body.action])
                        session.update(
                            observation=observations[0],
                            commands=list(infos["admissible_commands"][0]),
                            won=bool(infos["won"][0]),
                            steps=session["steps"] + 1,
                        )
                        session["done"] = bool(dones[0]) or session["won"] or session["steps"] >= max_steps
                    except Exception:
                        session.update(error=True, done=True)
                        raise
                return view(session)
            finally:
                # Refresh after the step so a long call cannot age the session
                # out while its lock is held (see purge_expired_sessions).
                session["last_used"] = time.time()

    @app.delete("/sessions/{session_id}")
    def close_session(session_id: str):
        with registry_lock:
            session = sessions.pop(session_id, None)
        if session is not None:
            with session["lock"]:
                with env_lock:
                    session["env"].close()
        return {"closed": True}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(os.environ["ALFWORLD_DATA"], int(os.environ.get("ALFWORLD_MAX_STEPS", "50"))),
        host="127.0.0.1",
        port=8765,
    )

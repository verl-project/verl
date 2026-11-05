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

import pytest

from verl.trainer.ppo.v1 import tq_schema


def test_keys_round_trip_to_owning_uid():
    uid = "0123abcd"
    session = tq_schema.session_key(uid, 2)
    trajectory = tq_schema.trajectory_key(uid, 2, 5)

    assert session == "0123abcd_2"
    assert trajectory == "0123abcd_2_5"
    assert {tq_schema.uid_of(key) for key in (uid, session, trajectory)} == {uid}
    assert tq_schema.parse_trajectory_key(trajectory) == (uid, 2, 5)


@pytest.mark.parametrize("key", ["0123abcd", "0123abcd_2"])
def test_parse_trajectory_key_rejects_non_trajectory_keys(key):
    assert tq_schema.parse_trajectory_key(key) is None


def test_tags_are_plain_and_classified():
    prompt = tq_schema.prompt_tag(tq_schema.PROMPT_PENDING, global_steps=3)
    session = tq_schema.session_tag(tq_schema.SESSION_FAILURE, session_id=1)

    # Plain str values keep TransferQueue checkpoints written by older versions loadable.
    assert prompt == {"is_prompt": True, "status": "pending", "global_steps": 3}
    assert session == {"is_session": True, "status": "failure", "session_id": 1}
    assert tq_schema.prompt_status_tag(tq_schema.PROMPT_FINISHED) == {"status": "finished"}

    assert tq_schema.is_prompt_tag(prompt) and not tq_schema.is_session_tag(prompt)
    assert tq_schema.is_session_tag(session) and not tq_schema.is_prompt_tag(session)
    trajectory_tag = {"status": "success", "global_steps": 3}
    assert not tq_schema.is_prompt_tag(trajectory_tag) and not tq_schema.is_session_tag(trajectory_tag)

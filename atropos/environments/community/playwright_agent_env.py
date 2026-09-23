from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

# Playwright is used for browser automation
from playwright.async_api import Browser, Page, async_playwright

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Gemini (google-genai) – optional, only imported when used to score a rollout
try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover – gemini is optional in most dev environments
    genai = None  # type: ignore
    genai_types = None  # type: ignore


class WebTask(TypedDict):
    """Simple structure describing a task the agent should solve on the web."""

    url: str
    goal_description: str  # natural language description of the objective
    success_criterion: str  # phrase we expect to appear somewhere (used as simple fallback if Gemini unavailable)


class PlaywrightAgentEnv(BaseEnv):
    """An environment that lets an LLM control a Playwright browser to accomplish a goal.

    Each episode:
    1. Opens the target URL in a fresh Playwright context that records video.
    2. Repeatedly shows the current screenshot to the LLM and asks for the next JSON-encoded action.
    3. Executes the action (e.g. click, type, navigate).
    4. Stops when the LLM replies with {"action": "finish"} or the step-limit is reached.
    5. Uses Google Gemini to evaluate whether the goal was achieved from the recorded video
       and computes a reward favouring both correctness and fewer steps.
    """

    name = "playwright_agent"
    name_config_cls = BaseEnvConfig

    # ---- configurable hyper-parameters ----
    max_steps: int = 10  # maximum browser actions per episode
    gemini_model_name: str = "gemini-2.5-pro-preview-05-06"

    # ---------- lifecycle hooks ----------
    async def setup(self) -> None:  # type: ignore[override]
        """Initialise Playwright and the list of tasks."""
        # 1. Launch Playwright once and keep it for the lifetime of the env
        self._playwright = await async_playwright().start()
        self._browser: Browser = await self._playwright.chromium.launch(headless=True)

        # 2. Load tasks from webvoyager_data.jsonl
        self._tasks: List[WebTask] = []
        try:
            with open("data/webvoyager_data.jsonl", "r") as f:
                for line in f:
                    if line.strip():
                        task_data = json.loads(line)
                        self._tasks.append(
                            {
                                "url": task_data["web"],
                                "goal_description": task_data["ques"],
                                # Using empty string for success_criterion as we're using Gemini to judge
                                "success_criterion": "",
                            }
                        )
            print(f"Loaded {len(self._tasks)} tasks from webvoyager_data.jsonl")
        except Exception as e:
            print(f"Error loading tasks from webvoyager_data.jsonl: {e}")
            # Fallback to a single example task if loading fails
            self._tasks = [
                {
                    "url": "https://example.com",
                    "goal_description": (
                        "Locate and open the link that contains the text 'More information'. "
                        "Then finish."
                    ),
                    "success_criterion": "More information",
                }
            ]

        self._iter = 0

        # Track if we're in development/test mode
        self._dev_mode = os.environ.get("PLAYWRIGHT_ENV_DEV_MODE", "0") == "1"
        if self._dev_mode:
            print("Running in development mode - no LLM will be used")

    async def teardown(self) -> None:  # type: ignore[override]
        if hasattr(self, "_browser"):
            await self._browser.close()
        if hasattr(self, "_playwright"):
            await self._playwright.stop()

    async def get_next_item(self) -> Item:  # type: ignore[override]
        """Return the next task specification."""
        task = self._tasks[self._iter % len(self._tasks)]
        self._iter += 1

        # The prompt given to the LLM before any browser interaction
        initial_prompt = (
            frozenset(
                {
                    "role": "user",
                    "content": (
                        f"You are an autonomous web-agent. Your goal is: "
                        f"{task['goal_description']}\n"
                        "You will be sent browser screenshots.\n"
                        "Reply with a JSON object describing the next action.\n\n"
                        "Allowed actions:\n"
                        "  navigate <url> – navigate the browser to <url>\n"
                        "  click <selector> – click the first element matching <selector>\n"
                        "  type <selector> <text> – type <text> into element <selector> and press Enter\n"
                        "  finish – if the goal is accomplished.\n\n"
                        'Example: {"action": "click", "selector": "text=More information"}'
                    ),
                }.items()
            ),
        )
        # Ground truth is unknown at this stage – Gemini will judge later; we keep success_criterion for fallback
        return (initial_prompt, task["success_criterion"], None)

    # ---------------------------------------------------------------------
    # core rollout – interacts with browser, builds messages & returns scores
    # ---------------------------------------------------------------------
    async def collect_trajectories(self, item: Item) -> Tuple[GameHistory | None, List[Item]]:  # type: ignore[override]
        prompt_frozenset, success_criterion, _ = (
            item  # we stored criterion in position 1
        )

        # Extract content string properly
        prompt_dict = dict(prompt_frozenset)
        content = prompt_dict.get("content", "")

        # Handle different content structures to extract goal description
        goal_description = ""
        if isinstance(content, str):
            goal_description = content
        elif isinstance(content, list):
            # If content is a list of message parts, extract text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    goal_description += part.get("text", "")
        elif isinstance(content, dict):
            # Try to extract text from a dict structure
            if "text" in content:
                goal_description = content["text"]

        # Ensure we have a string
        goal_description = str(goal_description)

        # If we couldn't extract a meaningful description, use a fallback
        if not goal_description:
            goal_description = "No goal description available"

        # 1. Create new context+page with video recording
        tmp_dir = Path(tempfile.mkdtemp(prefix="playwright_run_"))
        context = await self._browser.new_context(record_video_dir=str(tmp_dir))
        page: Page = await context.new_page()

        # Extract target URL from goal description heuristically (fallback)
        target_url = self._extract_first_url(goal_description) or "about:blank"
        if target_url != "about:blank":
            try:
                await page.goto(target_url, wait_until="load")
            except Exception:
                traceback.print_exc()

        messages_for_llm: List[dict] = [
            dict(prompt_frozenset)
        ]  # start conversation history

        steps_taken = 0
        finished = False
        screenshot_b64 = ""

        # In development mode, we'll just take one screenshot and finish
        if self._dev_mode:
            try:
                screenshot_bytes = await page.screenshot(full_page=True)
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                print(f"Development mode: took screenshot of {target_url}")
                finished = True
            except Exception as e:
                print(f"Development mode: error taking screenshot: {e}")
                screenshot_b64 = ""
            steps_taken = 1
        else:
            # Normal mode with LLM interaction
            while steps_taken < self.max_steps and not finished:
                # ---- 1. capture screenshot ----
                try:
                    screenshot_bytes = await page.screenshot(full_page=True)
                    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                except Exception:
                    screenshot_b64 = ""

                # ---- 2. Ask LLM for next action ----
                user_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Step {steps_taken}. Provide the next action as JSON.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_b64}",
                            },
                        },
                    ],
                }
                messages_for_llm.append(user_msg)

                llm_response = await self.server.chat_completion(
                    messages=messages_for_llm,
                    n=1,
                    max_tokens=256,
                    timeout=60,
                )

                assistant_content = llm_response.choices[0].message.content.strip()
                messages_for_llm.append(
                    {"role": "assistant", "content": assistant_content}
                )

                # ---- 3. Execute LLM-proposed action ----
                try:
                    action_dict = json.loads(assistant_content)
                    action_name = action_dict.get("action")
                except Exception:
                    # malformed JSON → give up this episode
                    break

                try:
                    if action_name == "finish":
                        finished = True
                    elif action_name == "navigate":
                        await page.goto(action_dict["url"], wait_until="load")
                    elif action_name == "click":
                        await page.locator(action_dict["selector"]).first.click()
                    elif action_name == "type":
                        await page.locator(action_dict["selector"]).first.fill(
                            action_dict["text"]
                        )
                        await page.keyboard.press("Enter")
                    else:
                        # unsupported → no-op
                        pass
                except Exception:
                    traceback.print_exc()

                steps_taken += 1

                # simple heuristic exit if success text present and Gemini not available
                if (
                    not finished
                    and success_criterion
                    and success_criterion.lower() in (await page.content()).lower()
                ):
                    finished = True

        # Finalise the Playwright context and obtain video path
        try:
            await context.close()
        except Exception:
            traceback.print_exc()
        video_path: Optional[str] = None
        try:
            video_rel = next(tmp_dir.glob("**/*.webm"))  # Playwright stores .webm
            video_path = str(video_rel)
        except StopIteration:
            pass

        # ---------------------------------------------------
        # Evaluate episode outcome – Gemini if available
        # ---------------------------------------------------
        success = (
            True
            if self._dev_mode
            else await self._judge_success_with_gemini(
                video_path, goal_description, success_criterion
            )
        )

        reward_value = self._compute_reward(success, steps_taken)

        # ----------------------------------
        # format for trainer (single sample)
        # ----------------------------------
        if self._dev_mode:
            # In dev mode, create minimal return structure without tokenizer
            scored_group = ScoredDataGroup(
                tokens=[[0]],  # minimal token placeholder
                masks=[[1]],  # minimal mask placeholder
                scores=[reward_value],
                images=[screenshot_b64],
            )
        else:
            out_dict = tokenize_for_trainer(self.tokenizer, tuple(messages_for_llm))
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            scored_group = ScoredDataGroup(
                tokens=[tokens],
                masks=[masks],
                scores=[reward_value],
                images=[screenshot_b64],
            )
        return scored_group, []

    # ------------------------------------------------------------------
    # helper methods
    # ------------------------------------------------------------------
    async def _judge_success_with_gemini(
        self,
        video_path: Optional[str],
        goal_description: str,
        success_criterion: str | None,
    ) -> bool:
        """Return True if the episode is deemed successful."""
        # Fallback cheap check – use DOM text in last page
        if genai is None or video_path is None or not Path(video_path).exists():
            # Very simple heuristic using success_criterion (already checked earlier)
            return success_criterion is None or success_criterion == ""

        try:
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            file_res = client.files.upload(file=video_path)
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_uri(
                            file_uri=file_res.uri,
                            mime_type=file_res.mime_type,
                        ),
                        genai_types.Part.from_text(
                            text=(
                                "Did the agent in this video succeed in the following task?\n"
                                f"Task: {goal_description}\n"
                                "Answer with a single word – Yes or No."
                            )
                        ),
                    ],
                )
            ]
            gen_config = genai_types.GenerateContentConfig(
                response_mime_type="text/plain"
            )
            # Gemini streaming → collect text
            text_chunks: List[str] = []
            for chunk in client.models.generate_content_stream(
                model=self.gemini_model_name,
                contents=contents,
                config=gen_config,
            ):
                if hasattr(chunk, "text"):
                    text_chunks.append(chunk.text)
            judgment = " ".join(text_chunks).strip().lower()
            return "yes" in judgment
        except Exception:
            traceback.print_exc()
            return False

    def _compute_reward(self, success: bool, steps_taken: int) -> float:
        if not success:
            return -1.0
        # successful – normalise by steps (fewer is better)
        return 1.0 + max(0, (self.max_steps - steps_taken)) / self.max_steps

    @staticmethod
    def _extract_first_url(text: str) -> Optional[str]:
        import re

        # Handle None or non-string type
        if text is None or not isinstance(text, str):
            return None

        match = re.search(r"https?://[\w/.:\-]+", text)
        return match.group(0) if match else None

    async def evaluate(self, *args, **kwargs) -> None:
        """
        Evaluation method required by BaseEnv.
        Called periodically during training to assess model performance.

        This environment doesn't use custom evaluation metrics.
        """
        return None

    # -----------------------------------------------------
    # mandatory classmethod for config initialisation
    # -----------------------------------------------------
    @classmethod
    def config_init(cls):  # type: ignore[override]
        base_config = BaseEnvConfig(
            wandb_name="playwright_agent",
            tokenizer_name="Qwen/Qwen-tokenizer",
            group_size=1,  # we only need one rollout per episode for this env
            use_wandb=True,
            max_num_workers=1,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=4,
            steps_per_eval=50,
            max_token_length=2048,
        )

        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen-tokenizer",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=64,
            )
        ]

        return base_config, server_configs


if __name__ == "__main__":
    import sys

    # Add simple development mode option
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        print("Starting PlaywrightAgentEnv in development mode...")
        os.environ["PLAYWRIGHT_ENV_DEV_MODE"] = "1"

        async def test_env():
            # Get the default configurations
            default_config, default_server_configs = PlaywrightAgentEnv.config_init()

            # Create the environment with the default configs
            env = PlaywrightAgentEnv(
                config=default_config,
                server_configs=default_server_configs,
                testing=True,
            )
            await env.setup()

            # Create a simple test item
            test_task = {
                "url": "https://example.com",
                "goal_description": "Locate and click on the 'More information' link.",
                "success_criterion": "More information",
            }

            # Create a simple prompt
            prompt_set = frozenset(
                {
                    "role": "user",
                    "content": f"Test goal: {test_task['goal_description']}",
                }.items()
            )

            item = (prompt_set, test_task["success_criterion"], None)

            try:
                result, _ = await env.collect_trajectories(item)
                print("Successfully ran test trajectory")
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error in test trajectory: {e}")
                traceback.print_exc()
            finally:
                await env.teardown()

        asyncio.run(test_env())
    else:
        # Run CLI helper if invoked directly (inherits from BaseEnv)
        PlaywrightAgentEnv.cli()

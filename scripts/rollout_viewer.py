# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import asyncio
import re
import traceback
from pathlib import Path
from typing import Annotated, Optional

import aiofiles

try:
    import ujson as json
except ImportError:
    import json
import typer
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, ProgressBar, Select, SelectionList, Static

INDEX_KEY = "__IDX"
FILE_SUFFIX = ".jsonl"


def check_textual_version():
    # check if textual version is equal to 0.52.1
    import textual
    from packaging.version import Version

    if Version(textual.__version__) != Version("0.52.1"):
        raise ImportError(f"Textual version {textual.__version__} is not supported, please pip install textual==0.52.1")


check_textual_version()


async def load_path(p: Path, data: dict, mask_strs: str, idx: int, pbar):
    samples = []
    async with aiofiles.open(p, encoding="utf-8") as f:
        async for line in f:
            d = json.loads(line)
            for k in d:
                if isinstance(d[k], str):
                    if mask_strs:
                        d[k] = re.sub(rf"{mask_strs}", "*", d[k])
                else:
                    d[k] = json.dumps(d[k], ensure_ascii=False, indent=4)

            d[INDEX_KEY] = len(samples)
            samples.append(d)
        data[idx] = {"samples": samples}

    print(f"path {p} loaded")
    pbar.advance(1)


async def load_dir(path: Path, data: dict[int, dict], pbar, mask_strs: str = ""):
    paths = list(path.glob(f"*{FILE_SUFFIX}"))
    paths = sorted(paths, key=lambda x: int(x.stem))

    tasks = [load_path(p, data, mask_strs, i, pbar) for i, p in enumerate(paths)]

    await asyncio.gather(*tasks)


class Highlighter(ReprHighlighter):
    highlights = ReprHighlighter.highlights + [
        r"(?P<tag_name>[][\<\>{}()\|（）【】\[\]=`])",
        r"\<\|(?P<tag_name>[\w\W]*?)\|\>",
    ]


def center_word_with_equals_exactly(word: str, total_length: int, char: str = "=") -> str:
    if len(word) > total_length:
        return word

    padding = total_length - len(word)
    left_pad = (padding) // 2
    right_pad = (padding + 1) // 2
    return char * left_pad + " " + word + " " + char * right_pad


def highlight_keyword(content: str, keyword: Optional[str]):
    if not keyword:
        return Text(content)
    text = Text()
    parts = content.split(keyword)
    for i, part in enumerate(parts):
        text.append(part, style=None)
        if i < len(parts) - 1:
            # text.append(keyword, style=Style(color="#d154d1", bgcolor="yellow", bold=True))
            text.append(keyword, style="on #8f51b5")
    return text


help_doc = """
⌨️   keybinds：

- `f/esc`: find/cancel
- `tab/←/→`: change focus
- `j/k`: page down/up
- `g/G`: scroll home/end
- `n/N`: next sample/step
- `p/P`: previous sample/step
- `s`: switch display mode
  - plain text
  - rich table

"""


class JsonLineViewer(App):
    BINDINGS = [
        ("left", "focus_previous", "Focus Previous"),
        ("right", "focus_next", "Focus Next"),
        ("s", "swith_render", "switch render"),
        # control
        ("n", "next_sample", "Next Sample"),
        ("N", "next_step", "Next Step"),
        ("p", "previous_sample", "Previous Sample"),
        ("P", "previous_step", "Previous Step"),
        # search
        ("f", "toggle_search", "find"),
        ("enter", "next_search", "find next"),
        ("escape", "cancel_search", "cancel find"),
        # scroll
        ("j", "page_down", "page down"),
        ("k", "page_up", "page up"),
        ("g", "page_home", "page home"),
        ("G", "page_end", "page end"),
    ]

    CSS = """

    Select:focus > SelectCurrent {
        border: tall #8f51b5;
    }
    Select.-expanded > SelectCurrent {
        border: tall #8f51b5;
    }
    #select-container {
        width: 15%;
        height: 100%;
        align: center top;
    }
    #search-container {
        height: 10%;
        align: center top;
    }
    #search-box {
        width: 50%;
    }
    #reqid-box {
        width: 50%;
    }
    """

    def __init__(self, step_num: int, data: dict[int, dict], pbar):
        super().__init__()
        self.step_num = step_num

        self.data = data
        self.render_table = False
        self.selected_step_index = 0
        self.selected_sample_index = 0
        self.pbar = pbar

        self.matches = []
        self.current_match_index = 0

        self.highlighter = Highlighter()

        first_samples = data[list(data.keys())[0]]["samples"]
        # Prepare the initial field filter list (all keys from the first sample)
        self.filter_fields = [(f, f, True) for f in first_samples[0].keys()]

        # Internal set used for fast membership checks when we add new fields on the fly.
        # We keep it here so that when new columns appear in later steps (e.g. `request_id`),
        # they can be added to the UI automatically without restarting the viewer.
        self._field_set: set[str] = set(first_samples[0].keys())
        self.sample_num = len(first_samples)

    def compose(self) -> ComposeResult:
        with Horizontal(id="search-container"):
            yield Input(placeholder="find something...", id="search-box")
            yield Input(placeholder="request id...", id="reqid-box")
            with Vertical(id="search-container2"):
                yield self.pbar
                yield Static("", id="search-status")

        with Horizontal():
            with Vertical(id="select-container"):
                yield Static("\n")
                yield Static(
                    renderable=Markdown(
                        help_doc,
                    ),
                    markup=False,
                )
                yield Static("\n")
                yield Select(
                    id="step-select",
                    value=0,
                    prompt="select step",
                    options=[("step: 1", 0)],
                    allow_blank=False,
                )
                yield Select(
                    id="sample-select",
                    value=0,
                    prompt="select sample",
                    options=[("sample: 1", 0)],
                    allow_blank=False,
                )
                yield Select(
                    id="sample-sort",
                    value=0,
                    prompt="排序",
                    options=[
                        ("sort", 0),
                        ("score asc", 1),
                        ("score desc", 2),
                    ],
                    allow_blank=False,
                )

                yield SelectionList[int](("Select ALL", 1, True), id="fields-select-all")
                with VerticalScroll(id="scroll-view2"):
                    yield SelectionList[str](*self.filter_fields, id="fields-select")
            with VerticalScroll(id="scroll-view"):
                yield Static(id="content", markup=False)

    async def on_mount(self) -> None:
        self.step_select = self.query_one("#step-select", Select)
        self.sample_select = self.query_one("#sample-select", Select)
        self.sample_sort = self.query_one("#sample-sort", Select)
        self.content_display = self.query_one("#content", Static)
        self.search_box = self.query_one("#search-box", Input)
        self.reqid_box = self.query_one("#reqid-box", Input)
        self.scroll_view = self.query_one("#scroll-view", VerticalScroll)
        self.search_status = self.query_one("#search-status", Static)
        self.fields_select = self.query_one("#fields-select", SelectionList)
        self.fields_select.border_title = "field filter"

        if self.data:
            self.step_select.set_options([(f"step: {i + 1}", i) for i in range(self.step_num)])
            self.sample_select.set_options([(f"sample: {i + 1}", i) for i in range(self.sample_num)])
            self.step_select.focus()
            await self.update_content()

    def update_result_options(self, offset: int = 0, sort_desc: Optional[bool] = None):
        options = []
        if isinstance(self.selected_step_index, int) and self.selected_step_index < len(self.data):
            if self.sample_num is None or sort_desc is not None:
                samples = self.data[self.selected_step_index].get("samples", [])
                if not samples:
                    self.selected_sample_index = offset
                    return
                if sort_desc is not None:
                    samples = sorted(
                        samples,
                        key=lambda x: x.get("score", x.get("score_1", 0)),
                        reverse=sort_desc,
                    )

                options = [(f"sample: {r[INDEX_KEY] + 1}", r[INDEX_KEY]) for r in samples]
                self.sample_select.set_options(options)
                self.sample_num = len(samples)

            if sort_desc is not None and options:
                self.selected_sample_index = options[0][1]
            else:
                self.selected_sample_index = offset

    async def update_content(self, search_keyword: Optional[str] = None):
        content = ""
        try:
            samples = self.data[self.selected_step_index].get("samples", [])
            content_dict_full = samples[self.selected_sample_index]

            # Dynamically track any NEW keys that appear and add them to the field filter.
            self._update_fields_select(content_dict_full.keys())

            # Apply field selection filter (only show selected fields)
            content_dict = {k: v for k, v in content_dict_full.items() if k in self.fields_select.selected}
            if self.render_table:
                content = Table("key", "value", show_lines=True)
                for k in content_dict:
                    v = content_dict[k]
                    v = f"{v}"
                    content.add_row(
                        k,
                        self.highlighter(highlight_keyword(v, search_keyword)),
                    )
            else:
                text = Text()
                for k in content_dict:
                    v = content_dict[k]
                    s = center_word_with_equals_exactly(k, 64) + f"\n{v}\n"
                    text.append(highlight_keyword(s, search_keyword))
                content = self.highlighter(text)
        except KeyError:
            content = f"Loading data asynchronously, progress: {len(self.data)}/{self.step_num} step"

        except Exception:
            content = self.highlighter(traceback.format_exc())

        self.content_display.update(content)

    # ---------------------------------------------------------------------
    # Request-ID jump logic
    # ---------------------------------------------------------------------

    @on(Input.Submitted, "#reqid-box")
    async def on_reqid_submitted(self, event: Input.Submitted) -> None:
        """Jump to the sample that has a matching `request_id`."""

        req_id_raw = event.value.strip()
        # Remove hyphens so search is tolerant to different id formats
        req_id = req_id_raw.replace("-", "")
        if not req_id:
            return

        found = False
        for step_idx, step_data in self.data.items():
            for sample in step_data.get("samples", []):
                sample_id = str(sample.get("request_id", ""))
                if sample_id.replace("-", "") == req_id:
                    # Update selected indices
                    self.selected_step_index = step_idx
                    self.step_select.value = step_idx

                    # Ensure sample list is updated and select sample
                    self.update_result_options(offset=sample[INDEX_KEY])
                    self.selected_sample_index = sample[INDEX_KEY]
                    self.sample_select.value = sample[INDEX_KEY]

                    await self._clear_search()
                    await self.update_content()

                    found = True
                    break
            if found:
                break

        if not found:
            self.search_status.update(Text(f"request_id '{req_id_raw}' not found", style="bold red"))
        else:
            # Keep the typed id in the input box so users see what was searched.
            pass

    # ---------------------------------------------------------------------
    # Helper: add new fields to SelectionList on-the-fly
    # ---------------------------------------------------------------------

    def _update_fields_select(self, keys):
        """Add any unseen *keys* to the field-selection widget so they can be toggled.

        The viewer is often launched with only the first step loaded. Later steps may
        introduce new columns (e.g. `request_id`). This helper ensures those fields
        become visible without requiring a restart.
        """
        # Ensure we have the widget (only after on_mount)
        if not hasattr(self, "fields_select"):
            return

        for k in keys:
            if k not in self._field_set:
                self._field_set.add(k)
                try:
                    # By default, new fields are selected so they appear immediately.
                    self.fields_select.add_option(k, k, selected=True)
                except Exception:
                    # Fallback for older textual versions where signature is different.
                    self.fields_select.add_option((k, k, True))

    @on(Select.Changed, "#step-select")
    async def step_changed(self, event):
        self.selected_step_index = event.value
        self.update_result_options()
        await self.update_content()

    @on(Select.Changed, "#sample-select")
    async def sample_changed(self, event):
        self.selected_sample_index = event.value
        await self._clear_search()
        await self.update_content()

    @on(Select.Changed, "#sample-sort")
    async def sort_changed(self, event):
        v = event.value
        self.update_result_options(sort_desc=None if v == 0 else False if v == 1 else True)
        await self.update_content()

    @on(SelectionList.SelectedChanged, "#fields-select")
    async def fields_changed(self, event):
        await self.update_content()

    @on(SelectionList.SelectedChanged, "#fields-select-all")
    async def fields_all_changed(self, event):
        s = self.query_one("#fields-select-all", SelectionList)
        if s.selected:
            self.fields_select.select_all()
        else:
            self.fields_select.deselect_all()

    def action_focus_previous(self):
        self.screen.focus_previous()

    def action_focus_next(self):
        self.screen.focus_next()

    async def action_next_step(self) -> None:
        self.selected_step_index += 1
        if self.selected_step_index >= self.step_num:
            self.selected_step_index = 0
        self.step_select.value = self.selected_step_index
        self.update_result_options()
        await self.update_content()

    async def action_next_sample(self) -> None:
        self.selected_sample_index += 1
        if not self.sample_num or self.selected_sample_index >= self.sample_num:
            self.selected_sample_index = 0
        self.sample_select.value = self.selected_sample_index
        await self._clear_search()
        await self.update_content()

    async def action_previous_step(self) -> None:
        self.selected_step_index -= 1
        if self.selected_step_index < 0:
            self.selected_step_index = self.step_num - 1
        self.step_select.value = self.selected_step_index
        self.update_result_options()
        await self.update_content()

    async def action_previous_sample(self) -> None:
        self.selected_sample_index -= 1
        if self.selected_sample_index < 0:
            self.selected_sample_index = self.sample_num - 1
        self.sample_select.value = self.selected_sample_index
        await self._clear_search()
        await self.update_content()

    async def action_swith_render(self):
        self.render_table = not self.render_table
        await self.update_content()

    def action_toggle_search(self) -> None:
        self.search_box.focus()

    async def action_cancel_search(self) -> None:
        self.search_box.value = ""
        await self._clear_search()
        await self.update_content()

    async def _clear_search(self):
        self.matches = []
        self.search_status.update("")
        self.current_match_index = 0

    @on(Input.Submitted, "#search-box")
    async def on_search_submitted(self, event: Input.Submitted) -> None:
        self.matches = []
        self.current_match_index = 0
        if event.value:
            await self.update_content(event.value)
            renderable = self.content_display.render()
            if isinstance(renderable, Table):
                return

            assert isinstance(renderable, Text)
            console = self.content_display._console
            lines = renderable.wrap(console, self.scroll_view.container_size.width)
            line_idx_recorded = set()
            for line_idx, line in enumerate(lines):
                if line_idx in line_idx_recorded:
                    continue
                if event.value in line:
                    self.matches.append(
                        {
                            "line": line_idx,
                            "word": event.value,
                        }
                    )
                    line_idx_recorded.add(line_idx)
            self.scroll_view.focus()
            await self.action_next_search()

    async def action_next_search(self) -> None:
        if not self.matches or self.current_match_index >= len(self.matches):
            return

        target_line = self.matches[self.current_match_index]["line"]
        self.scroll_view.scroll_to(x=0, y=target_line * 1, animate=False)
        self.current_match_index = (self.current_match_index + 1) % len(self.matches)
        self.search_status.update(
            Text(
                f"Find ：{self.current_match_index + 1}/{len(self.matches)}",
                style="bold on #8f51b5",
            )
        )

    def action_page_up(self):
        self.scroll_view.scroll_page_up(animate=False)

    def action_page_down(self):
        self.scroll_view.scroll_page_down(animate=False)

    def action_page_home(self):
        self.scroll_view.scroll_home(animate=False)

    def action_page_end(self):
        self.scroll_view.scroll_end(animate=False)


async def _run(path: Path, mask_str: str):
    assert path.exists(), f"{path} not exist"

    paths = list(path.glob(f"*{FILE_SUFFIX}"))
    paths = sorted(paths, key=lambda x: int(x.stem))

    if not paths:
        raise ValueError(f"no available reward dump files under f{path}")

    print(f"get jsonl file nums: {len(paths)}")

    pbar = ProgressBar(total=len(paths), name="data load progress")
    data = {}
    await load_path(paths[0], data, mask_str, 0, pbar)
    app = JsonLineViewer(step_num=len(paths), data=data, pbar=pbar)
    await asyncio.gather(load_dir(path, data, pbar, mask_str), app.run_async())


app = typer.Typer()


@app.command(help="launch TUI APP")
def run(
    rollout_data_dir: Path,
    mask_str: Annotated[str, typer.Option(help="string that will be masked to *")] = r"<\|image_pad\|>|<\|imgpad\|>",
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run(rollout_data_dir, mask_str))


if __name__ == "__main__":
    app()

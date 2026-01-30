# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Utils for IO"""

import csv
import json
import logging
import os
import pathlib
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)


def read_lines(filename: str | pathlib.Path, skip_header: bool = False) -> list[str]:
    """Read lines from the filename into a list and optionally skip the header"""
    with open(filename, encoding="utf-8") as fid:
        if skip_header:
            fid.readline()  # skip the header
        lines = fid.read().splitlines()
    result = [line.strip() for line in lines]
    return [x for x in result if x]


def load_jsonl(filename: str | pathlib.Path) -> list[dict]:
    """Loads jsonl file into a list of Dict objects"""
    lines = read_lines(filename)
    return [json.loads(line) for line in lines]


def load_json(filename: str | pathlib.Path) -> dict:
    """Load the text file as a json doc"""
    with open(filename, encoding="utf-8") as fid:
        return json.load(fid)


def save_text(text: str, filename: str | pathlib.Path) -> None:
    """Save a list of json docs into jsonl file"""
    with open(filename, "w", encoding="utf-8") as fid:
        fid.write(text)


def save_jsonl(docs: list[dict], filename: str | pathlib.Path, mode: Literal["a", "w"] = "w") -> None:
    """Write/Append a list of json docs into jsonl file"""
    if mode not in ["a", "w"]:
        raise AttributeError('mode needs to be one of ["a", "w"]')
    with open(filename, mode, encoding="utf-8") as fid:
        for doc in docs:
            line = json.dumps(doc)
            fid.write(line + "\n")


def save_json(doc: dict, filename: str | pathlib.Path) -> None:
    """Load the text file as a json doc"""
    with open(filename, "w", encoding="utf-8") as fid:
        json.dump(doc.copy(), fid, indent=2)


def write_list_to_file(src_list: list[Any], filename: pathlib.Path | str) -> None:
    """Write lines into text file"""
    filename = str(filename)
    with open(filename, "w") as fh:
        for v in src_list:
            fh.write(f"{v}\n")


def write_text_to_file(text: str, filename: pathlib.Path | str) -> None:
    """Write lines into text file"""
    filename = str(filename)
    create_dir_if_not_exist(filename)
    with open(filename, "w") as fh:
        fh.write(text)


def save_csv(
    contents: list[dict[str, Any]],
    columns: list[str],
    filename: pathlib.Path | str,
) -> None:
    """Save a list of key value pairs into csv file"""
    assert len(columns) > 0
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in contents:
            writer.writerow(row)


def save_csv_columns(contents: dict[str, list], filename: pathlib.Path | str, log: bool = False):
    """Save columnar content to a csv file
    contents format:
    {
        "col_a": [1,2,3],
        "col_b": ['a', 'b', 'c']
    }
    """
    df = pd.DataFrame(contents)
    logger.info(f"Save data with {len(df)} rows and {len(df.columns)} columns to csv file {filename}")
    if log:
        logger.info(f"The contents are:\n{df.to_string(index=False)}")
    df.to_csv(filename, index=False)


def create_dir_if_not_exist(filename: str):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(f"{dir_path}"):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

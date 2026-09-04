#!/usr/bin/env python3
"""
checker_gpt5mini.py — GPT-5-mini checker server (port 8005)

Drop-in replacement for checker_medrag_gpt4omini.py using gpt-5-mini.
Runs on port 8005 to coexist with the existing gpt-4o-mini server on 8004.

Usage:
    # Start server (background)
    nohup python search_r1_preprocess/checker_gpt5mini.py \
        --mode openai \
        --port 8005 \
        > checker_gpt5mini.log 2>&1 &

    # Health check
    curl -s http://127.0.0.1:8005/health
"""

import sys, os

# Reuse the original checker code, just override defaults
_orig = "/ocean/projects/med230010p/yji3/BrowseCamp/verl/search_r1_preprocess/checker_medrag_gpt4omini.py"
sys.path.insert(0, os.path.dirname(_orig))

# Patch sys.argv defaults before importing
import importlib.util

# Load original module source
with open(_orig) as f:
    src = f.read()

# Replace default model name and port in source
src = src.replace(
    'model_name: str = "gpt-4o-mini"',
    'model_name: str = "gpt-5-mini"'
).replace(
    '"gpt-4o-mini"',
    '"gpt-5-mini"'
).replace(
    'default="gpt-4o-mini"',
    'default="gpt-5-mini"'
).replace(
    '--port", type=int, default=8004',
    '--port", type=int, default=8005'
).replace(
    'default=8004',
    'default=8005'
)

# Execute patched source
exec(compile(src, _orig, 'exec'), {'__name__': '__main__', '__file__': _orig})

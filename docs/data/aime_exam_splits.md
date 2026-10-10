# AIME validation by exam session

Last updated: 10/04/2026.

`examples/data_preprocess/aime_exam_splits.py` prepares separate AIME I/II validation files for 2024, 2025, and 2026. Each file contains 15 problems; the combined file contains 90. Distinct `data_source` values such as `aime2026-I` preserve exam identity in validation metrics.

```bash
python examples/data_preprocess/aime_exam_splits.py --output-dir /tmp/aime-exams
```

Dataset revisions, expected file sizes and SHA-256 hashes are pinned. The generator verifies question membership and answers against public exam-specific sources, checks Parquet round trips, and writes a provenance manifest. Optional `--existing-aime2024` and `--existing-aime2025` files preserve existing prompts and metadata after exact source verification. `--source-dir` accepts an offline source cache whose files still undergo size/hash checks.

The 2026 II data preserves the publisher's regional problem variant and answer together; the manifest explains the source mapping. Preserve the dataset license and attribution recorded in the source manifest when using the generated data. No training data or model weights are downloaded.

MathArena sources declare CC BY-NC-SA 4.0. The pinned HuggingFaceH4 AIME 2024 card does not declare a license; the manifest records this as unknown and links the pinned card. Do not infer that MathArena’s license covers the H4 text, or that the generated files inherit the generator’s Apache license. Check each source’s usage rights independently.

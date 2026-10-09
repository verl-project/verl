# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Token-weighted generation ages from compact partial-rollout segments."""


def behavior_age_metrics(extra_fields, response_lengths, non_padding, current_version):
    """Report coverage explicitly; missing provenance is never counted as age zero.

    Each segment is [generation_version, generated_token_count]. Only complete
    response provenance contributes to age and cross-version statistics.
    """
    if type(current_version) is not int or current_version < 0:
        raise ValueError("current_version must be a nonnegative integer")
    ages = {}
    eligible = covered = cross_version = expected_tokens = covered_tokens = 0
    for extra, length, active in zip(extra_fields, response_lengths, non_padding, strict=True):
        if not active:
            continue
        eligible += 1
        expected_tokens += length
        segments = extra.get("behavior_version_segments") if isinstance(extra, dict) else None
        if (
            not isinstance(segments, list | tuple)
            or not segments
            or not all(
                isinstance(s, list | tuple)
                and len(s) == 2
                and type(s[0]) is int
                and 0 <= s[0] <= current_version
                and type(s[1]) is int
                and s[1] > 0
                for s in segments
            )
        ):
            continue
        if sum(s[1] for s in segments) != length:
            continue
        covered += 1
        covered_tokens += length
        cross_version += len({s[0] for s in segments}) > 1
        for version, count in segments:
            age = current_version - version
            ages[age] = ages.get(age, 0) + count
    prefix = "training/off_policy/"
    result = {
        prefix + "behavior_version/response_coverage": covered / eligible if eligible else 0.0,
        prefix + "behavior_version/token_coverage": covered_tokens / expected_tokens if expected_tokens else 0.0,
    }
    if covered_tokens:
        result.update(
            {
                prefix + "token_staleness/mean": sum(age * count for age, count in ages.items()) / covered_tokens,
                prefix + "token_staleness/max": max(ages),
                prefix + "token_staleness/min": min(ages),
                prefix + "token_staleness/stale_fraction": sum(n for age, n in ages.items() if age > 0)
                / covered_tokens,
                prefix + "behavior_version/cross_version_response_fraction": cross_version / covered,
            }
        )
    return result


def read_behavior_version_fields(keys, partition_id, kv_batch_get):
    """Read optional provenance, retaining coverage for rows from older runs."""
    if not keys:
        return []
    try:
        data = kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["extra_fields"])
        values = data.get("extra_fields")
        return values.tolist() if values is not None else [None] * len(keys)
    except ValueError as exc:
        if "extra_fields" not in str(exc) and str(exc) != "Some fields are not ready in all the requested keys!":
            raise
    # A missing field in one row must not hide provenance in the other rows.
    result = []
    for key in keys:
        try:
            data = kv_batch_get(keys=[key], partition_id=partition_id, select_fields=["extra_fields"])
            values = data.get("extra_fields")
            result.append(values.tolist()[0] if values is not None else None)
        except ValueError as exc:
            if "extra_fields" not in str(exc) and str(exc) != "Some fields are not ready in all the requested keys!":
                raise
            result.append(None)
    return result

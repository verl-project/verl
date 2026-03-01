from typing import Any


def unique(xs: list) -> list:
    """Returns deduplicated list"""
    res = []
    for x in xs:
        if x not in res:
            res.append(x)
    return res


def grid_search(d: dict[str, Any]) -> list[dict[str, Any]]:
    ret = []
    if isinstance(d, dict):
        if len(d) == 0:
            return [dict()]
        copy = d.copy()
        k, v = copy.popitem()
        configs = unique(grid_search(v))
        rest = grid_search(copy)
        for r in rest:
            for config in configs:
                ret.append(r | {k: config})
    elif isinstance(d, list):
        for v in d:
            configs = grid_search(v)
            for config in configs:
                ret.append(config)
    else:
        ret.append(d)
    return ret

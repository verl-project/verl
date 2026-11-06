"""Accelerator accounting for resource pools."""


def accelerator_total(process_on_nodes: dict[str, list[int]], accelerators_per_process: dict[str, int]) -> int:
    """Count accelerators when one process may own more than one device.

    ``process_on_nodes`` maps a pool name to the process count on each node.
    ``accelerators_per_process`` multiplies a pool whose processes each hold
    several devices, which is how a single-controller JAX actor owns a node.
    """
    return sum(sum(nodes) * accelerators_per_process.get(name, 1) for name, nodes in process_on_nodes.items())

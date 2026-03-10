import ray
import os
import json
from ray.util.placement_group import PlacementGroup, placement_group

import tree_topology

def topology_aware_schedule(pgs, strategy, pg_name_prefix, lifetime):
    acquire_node_num = len(pgs)

    tr = build_tree_topo()

    try:
        schedule_list = tr.schedule(acquire_node_num)
    except Exception as e:
        print(e)
        return

    pg_scheme = []
    for index, pg in enumerate(pgs):
        bundles = pg.bundle_specs
        for bundle in bundles:
            bundle[schedule_list[index]] = 0.01
        pg_scheme.append(bundles)

    result = [
        placement_group(
            bundles,
            strategy=strategy,
            lifetime=lifetime,
            name=pg_name_prefix + str(index)
        ) for index, bundles in enumerate(pg_scheme)
    ]

    return result

def build_tree_topo():
    node_mapping, height = get_node_info()

    tr = tree_topology.TreeTopology(height)

    for _, info in node_mapping.items():
        tr.insert_node(info["Labels"], info["Blocks"])

    tr.set_strategy(tr.root)
    tr.update_topology(tr.root)

    return tr

def get_node_info():
    label_dict = {}
    node_details = ray.nodes()
    height = 0

    for i, node in enumberate(node_details):
        labels_dict = node.get("Labels", {})
        node_id = labels_dict.get("L0", "")

        l_labels = []
        for key, value in labels_dict.items():
            if key.startswith("L") and len(key) > 1:
                try:
                    n = int(key[1:])
                    l_labels.append((n, value))
                except ValueError:
                    continue

        l_labels.sort(key=lambda x: x[0], reverse=True)

        labels_dict = node.get("Labels", {})
        node_id = labels_dict.get("L0", "")

        l_labels = []
        for key, value in labels_dict.items():
            if key.startswith("L") and len(key) > 1:
                try:
                    n = int(key[1:])
                    value = value.split(":")
                    l_labels.append((n, value[0]))
                    l_blocks.append((n, int(value[1])))
                    height = max(height, n)
                except ValueError:
                    continue

        l_labels.sort(key=lambda x: x[0], reverse=True)
        l_blocks.sort(key=lambda x: x[0], reverse=True)

        label_dict[node_id] = {
            "Labels": [value for _, value in l_labels],
            "Blocks": [value for _, value in l_blocks],
        }

    return label_dict, height + 1
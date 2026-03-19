# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import logger
import ray
from ray.util.placement_group import placement_group

from .tree_topology import TreeTopology


def topology_aware_schedule(pgs, strategy, pg_name_prefix, lifetime):
    acquire_node_num = len(pgs)

    tr = build_tree_topo()

    if tr is None:
        return pgs

    try:
        schedule_list = tr.schedule(acquire_node_num)
    except Exception as e:
        logger.exception("Topology-aware scheduling failed, falling back to default scheduling: %s", e)
        return pgs

    pg_scheme = []
    for index, pg in enumerate(pgs):
        bundles = pg.bundle_specs
        for bundle in bundles:
            bundle[schedule_list[index]] = 0.01
        pg_scheme.append(bundles)

    result = [
        placement_group(
            bundles, strategy=strategy, lifetime=lifetime, name=pg_name_prefix + "_npu_topology" + str(index)
        )
        for index, bundles in enumerate(pg_scheme)
    ]

    return result


def build_tree_topo():
    node_mapping, height = get_node_info()

    if not node_mapping or height == 0:
        return None

    tr = TreeTopology(height)

    for _, info in node_mapping.items():
        tr.insert_node(info["Labels"], info["Blocks"])

    tr.set_strategy(tr.root)
    tr.update_topology(tr.root)

    return tr


def get_node_info():
    label_dict = {}
    node_details = ray.nodes()
    height = 0
    has_l0 = False

    for i, node in enumerate(node_details):
        labels_dict = node.get("Labels", {})
        node_id = labels_dict.get("L0", "")

        if node_id and ":" in node_id:
            has_l0 = True

        l_labels = []
        l_blocks = []
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

    if not has_l0:
        return {}, 0

    return label_dict, height + 1

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.single_controller.ray.base import _ip_sort_key


def test_mixed_width_ips_sort_numerically():
    ips = ["10.0.0.7", "10.0.0.66", "10.0.0.8", "10.0.0.70"]
    assert sorted(ips, key=_ip_sort_key) == ["10.0.0.7", "10.0.0.8", "10.0.0.66", "10.0.0.70"]


def test_equal_width_order_is_byte_identical_to_lexicographic():
    ips = ["10.0.0.101", "10.0.0.100", "10.0.0.103", "10.0.0.102"]
    assert sorted(ips, key=_ip_sort_key) == sorted(ips)


def test_full_octet_boundaries():
    ips = ["192.168.1.5", "9.0.0.1", "10.42.13.2", "10.42.9.66", "10.42.9.7"]
    assert sorted(ips, key=_ip_sort_key) == [
        "9.0.0.1",
        "10.42.9.7",
        "10.42.9.66",
        "10.42.13.2",
        "192.168.1.5",
    ]


def test_non_ipv4_sorts_after_ipv4_and_keeps_string_order():
    mixed = ["nodeb", "10.0.0.9", "nodea", "10.0.0.10", "fe80::1"]
    assert sorted(mixed, key=_ip_sort_key) == ["10.0.0.9", "10.0.0.10", "fe80::1", "nodea", "nodeb"]


def test_malformed_dotted_quad_falls_back_to_string():
    vals = ["10.0.0.x", "10.0.0.2"]
    assert sorted(vals, key=_ip_sort_key) == ["10.0.0.2", "10.0.0.x"]

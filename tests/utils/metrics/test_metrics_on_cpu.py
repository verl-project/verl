import unittest

import numpy as np
import pytest
import torch

from verl.utils.metric.metric import MetricAggregationType, MetricConfig, Metrics


class TestMetricConfig(unittest.TestCase):
    def test_default_config(self):
        config = MetricConfig("test_metric")
        assert config.name == "test_metric"
        assert config.aggregation == MetricAggregationType.MEAN
        assert config.format_str == "{:.3f}"

    def test_custom_config(self):
        config = MetricConfig(name="custom_metric", aggregation=MetricAggregationType.MAX, format_str="{:.6f}")
        assert config.name == "custom_metric"
        assert config.aggregation == MetricAggregationType.MAX
        assert config.format_str == "{:.6f}"

    def test_config_equality(self):
        config1 = MetricConfig("test", MetricAggregationType.MEAN, "{:.3f}")
        config2 = MetricConfig("test", MetricAggregationType.MEAN, "{:.3f}")
        config3 = MetricConfig("test", MetricAggregationType.MAX, "{:.3f}")

        assert config1 == config2
        assert config1 != config3


class TestMetrics(unittest.TestCase):
    def test_init(self):
        metrics = Metrics()
        assert isinstance(metrics.data, dict)
        assert isinstance(metrics.configs, dict)
        assert len(metrics.data) == 0
        assert len(metrics.configs) == 0

    def test_add_basic_values(self):
        metrics = Metrics()

        metrics.add("int_metric", 10)
        assert metrics.data["int_metric"] == [10]
        assert metrics.configs["int_metric"].aggregation == MetricAggregationType.MEAN
        assert metrics.configs["int_metric"].format_str == "{:.3f}"

        metrics.add("float_metric", 3.14)
        assert metrics.data["float_metric"] == [3.14]
        assert metrics.configs["float_metric"] == MetricConfig("float_metric", MetricAggregationType.MEAN, "{:.3f}")

        metrics.add("int_metric", 20)
        assert metrics.data["int_metric"] == [10, 20]
        assert metrics.configs["int_metric"] == MetricConfig("int_metric", MetricAggregationType.MEAN, "{:.3f}")

    def test_add_torch_tensors(self):
        metrics = Metrics()

        scalar_tensor = torch.tensor(5.0)
        metrics.add("tensor_metric", scalar_tensor)
        assert metrics.data["tensor_metric"] == [5.0]
        assert metrics.configs["tensor_metric"] == MetricConfig("tensor_metric", MetricAggregationType.MEAN, "{:.3f}")

        tensor_with_grad = torch.tensor(2.5, requires_grad=True)
        metrics.add("grad_tensor", tensor_with_grad)
        assert metrics.data["grad_tensor"] == [2.5]
        assert metrics.configs["grad_tensor"] == MetricConfig("grad_tensor", MetricAggregationType.MEAN, "{:.3f}")

    def test_add_numpy_arrays(self):
        metrics = Metrics()

        scalar_array = np.array(7.5)
        metrics.add("numpy_metric", scalar_array)
        assert metrics.data["numpy_metric"] == [7.5]
        assert metrics.configs["numpy_metric"] == MetricConfig("numpy_metric", MetricAggregationType.MEAN, "{:.3f}")

        int_array = np.array(42, dtype=np.int32)
        metrics.add("numpy_int", int_array)
        assert metrics.data["numpy_int"] == [42]
        assert metrics.configs["numpy_int"] == MetricConfig("numpy_int", MetricAggregationType.MEAN, "{:.3f}")

    def test_add_non_scalar_tensors_error(self):
        metrics = Metrics()

        # Test 1D tensor
        with pytest.raises(ValueError, match="must be a scalar"):
            metrics.add("bad_tensor", torch.tensor([1, 2, 3]))

        # Test 2D array
        with pytest.raises(ValueError, match="must be a scalar"):
            metrics.add("bad_array", np.array([[1, 2], [3, 4]]))

    def test_add_with_custom_config(self):
        metrics = Metrics()
        config = MetricConfig("custom", MetricAggregationType.MAX, "{:.6f}")

        metrics.add("custom_metric", 1.5, config=config)
        assert metrics.configs["custom_metric"] == config

    def test_add_config_override(self):
        """Test that config can be overridden for existing metrics."""
        metrics = Metrics()

        # Add with default config
        metrics.add("metric", 1.0)
        original_config = metrics.configs["metric"]

        # Override config
        new_config = MetricConfig("metric", MetricAggregationType.SUM, "{:.1f}")
        metrics.add("metric", 2.0, config=new_config)

        assert metrics.configs["metric"].aggregation == MetricAggregationType.SUM
        assert metrics.configs["metric"].format_str == "{:.1f}"
        assert metrics.configs["metric"] == new_config
        assert metrics.configs["metric"] != original_config

    def test_auto_infer_config_sum(self):
        metrics = Metrics()

        test_names = ["total_sum", "sum_value", "SUM_METRIC"]
        for name in test_names:
            config = metrics._auto_infer_config(name)
            assert config.aggregation == MetricAggregationType.SUM
            assert config.format_str == "{:.3f}"

    def test_auto_infer_config_max(self):
        metrics = Metrics()

        test_names = ["max_value", "peak_performance", "MAX_SCORE"]
        for name in test_names:
            config = metrics._auto_infer_config(name)
            assert config.aggregation == MetricAggregationType.MAX

    def test_auto_infer_config_min(self):
        metrics = Metrics()

        test_names = ["min_loss", "lowest_score", "MIN_VALUE"]
        for name in test_names:
            config = metrics._auto_infer_config(name)
            assert config.aggregation == MetricAggregationType.MIN

    def test_auto_infer_config_lr(self):
        metrics = Metrics()

        test_names = ["lr", "learning_rate", "actor_lr", "LR_VALUE"]
        for name in test_names:
            config = metrics._auto_infer_config(name)
            assert config.format_str == "{:.6f}"

    def test_auto_infer_config_default(self):
        metrics = Metrics()

        config = metrics._auto_infer_config("random_metric")
        assert config.aggregation == MetricAggregationType.MEAN
        assert config.format_str == "{:.3f}"

    def test_from_dict_single_values(self):
        data = {"loss": 0.5, "accuracy": 0.95, "lr": 1e-4}

        metrics = Metrics.from_dict(data)

        assert metrics.data["loss"] == [0.5]
        assert metrics.data["accuracy"] == [0.95]
        assert metrics.data["lr"] == [1e-4]
        assert len(metrics.configs) == 3
        assert metrics.configs["loss"] == MetricConfig("loss", MetricAggregationType.MEAN, "{:.3f}")
        assert metrics.configs["accuracy"] == MetricConfig("accuracy", MetricAggregationType.MEAN, "{:.3f}")
        assert metrics.configs["lr"] == MetricConfig("lr", MetricAggregationType.MEAN, "{:.6f}")

    def test_from_dict_list_values(self):
        data = {"loss": [0.5, 0.4, 0.3], "accuracy": [0.9, 0.95, 0.92], "lr": [1e-4, 1e-5, 1e-6]}

        metrics = Metrics.from_dict(data)

        assert metrics.data["loss"] == [0.5, 0.4, 0.3]
        assert metrics.data["accuracy"] == [0.9, 0.95, 0.92]
        assert metrics.data["lr"] == [1e-4, 1e-5, 1e-6]
        assert metrics.configs["loss"] == MetricConfig("loss", MetricAggregationType.MEAN, "{:.3f}")
        assert metrics.configs["accuracy"] == MetricConfig("accuracy", MetricAggregationType.MEAN, "{:.3f}")
        assert metrics.configs["lr"] == MetricConfig("lr", MetricAggregationType.MEAN, "{:.6f}")

    def test_from_dict_mixed_values(self):
        data = {"single_value": 1.0, "list_values": [1.0, 2.0, 3.0], "tuple_values": (4.0, 5.0)}

        metrics = Metrics.from_dict(data)

        assert metrics.data["single_value"] == [1.0]
        assert metrics.data["list_values"] == [1.0, 2.0, 3.0]
        assert metrics.data["tuple_values"] == [4.0, 5.0]
        assert metrics.configs["single_value"] == MetricConfig("single_value", MetricAggregationType.MEAN, "{:.3f}")
        assert metrics.configs["list_values"] == MetricConfig("list_values", MetricAggregationType.MEAN, "{:.3f}")
        assert metrics.configs["tuple_values"] == MetricConfig("tuple_values", MetricAggregationType.MEAN, "{:.3f}")

    def test_from_dict_with_configs(self):
        data = {"metric": 1.0}
        configs = {"metric": MetricConfig("metric", MetricAggregationType.MAX, "{:.1f}")}

        metrics = Metrics.from_dict(data, configs)
        assert metrics.data["metric"] == [1.0]
        assert metrics.configs["metric"] == MetricConfig("metric", MetricAggregationType.MAX, "{:.1f}")
        assert metrics.configs["metric"] == configs["metric"]

    def test_from_dict_deep_copy(self):
        original_data = {"metric": [1.0, 2.0]}
        metrics = Metrics.from_dict(original_data)

        original_data["metric"].append(3.0)

        assert metrics.data["metric"] == [1.0, 2.0]

    def test_merge_new_metrics(self):
        metrics1 = Metrics()
        metrics1.add("loss", 0.5)

        metrics2 = Metrics()
        metrics2.add("accuracy", 0.9)

        result = metrics1.merge(metrics2)

        assert result is metrics1
        assert "loss" in metrics1
        assert "accuracy" in metrics1
        assert metrics1.data["loss"] == [0.5]
        assert metrics1.data["accuracy"] == [0.9]

    def test_merge_existing_metrics(self):
        metrics1 = Metrics()
        metrics1.add("loss", 0.5)

        metrics2 = Metrics()
        metrics2.add("loss", 0.4)
        metrics2.add("loss", 0.3)

        metrics1.merge(metrics2)

        assert metrics1.data["loss"] == [0.5, 0.4, 0.3]

    def test_merge_from_dict(self):
        metrics = Metrics()
        metrics.add("existing", 1.0)

        data_dict = {"new_metric": 2.0, "existing": 3.0}
        metrics.merge(data_dict)  # type: ignore

        assert metrics.data["new_metric"] == [2.0]
        assert metrics.data["existing"] == [1.0, 3.0]

    def test_merge_strict_config_success(self):
        config = MetricConfig("test", MetricAggregationType.MAX)

        metrics1 = Metrics()
        metrics1.add("test", 1.0, config)

        metrics2 = Metrics()
        metrics2.add("test", 2.0, config)

        metrics1.merge(metrics2, strict_config=True)
        assert metrics1.data["test"] == [1.0, 2.0]
        assert metrics1.configs["test"] == config

    def test_merge_strict_config_failure(self):
        config1 = MetricConfig("test", MetricAggregationType.MEAN)
        config2 = MetricConfig("test", MetricAggregationType.MAX)

        metrics1 = Metrics()
        metrics1.add("test", 1.0, config1)

        metrics2 = Metrics()
        metrics2.add("test", 2.0, config2)

        with pytest.raises(ValueError, match="Config for metric test is different"):
            metrics1.merge(metrics2, strict_config=True)

    def test_merge_invalid_type(self):
        metrics = Metrics()

        with pytest.raises(TypeError, match="Expected Metrics or dict"):
            metrics.merge("invalid")  # type: ignore

    def test_update_new_metrics(self):
        metrics1 = Metrics()
        metrics1.add("loss", 0.5)

        metrics2 = Metrics()
        metrics2.add("accuracy", 0.9)

        result = metrics1.update(metrics2)

        assert result is metrics1
        assert "loss" in metrics1.data
        assert "accuracy" in metrics1.data
        assert metrics1.data["accuracy"] == [0.9]
        assert metrics1.data["loss"] == [0.5]

    def test_update_existing_metrics(self):
        metrics1 = Metrics()
        metrics1.add("loss", 0.5)
        metrics1.add("loss", 0.4)

        metrics2 = Metrics()
        metrics2.add("loss", 0.3)
        metrics2.add("loss", 0.2)

        metrics1.update(metrics2)

        # Data should be replaced, not extended
        assert metrics1.data["loss"] == [0.3, 0.2]

    def test_update_from_dict(self):
        """Test updating from dictionary."""
        metrics = Metrics()
        metrics.add("existing", 1.0)

        data_dict = {"new_metric": 2.0, "existing": 3.0}
        metrics.update(data_dict)

        assert metrics.data["new_metric"] == [2.0]
        assert metrics.data["existing"] == [3.0]

    def test_aggregate_value_mean(self):
        metrics = Metrics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = metrics._aggregate_value(values, MetricAggregationType.MEAN)
        assert result == 3.0

    def test_aggregate_value_sum(self):
        metrics = Metrics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = metrics._aggregate_value(values, MetricAggregationType.SUM)
        assert result == 15.0

    def test_aggregate_value_max(self):
        metrics = Metrics()
        values = [1.0, 5.0, 3.0, 2.0, 4.0]

        result = metrics._aggregate_value(values, MetricAggregationType.MAX)
        assert result == 5.0

    def test_aggregate_value_min(self):
        metrics = Metrics()
        values = [5.0, 1.0, 3.0, 2.0, 4.0]

        result = metrics._aggregate_value(values, MetricAggregationType.MIN)
        assert result == 1.0

    def test_aggregate_value_std(self):
        metrics = Metrics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = metrics._aggregate_value(values, MetricAggregationType.STD)
        expected = np.std(values)
        assert abs(result - expected) < 1e-10

    def test_aggregate_value_median(self):
        metrics = Metrics()

        values_odd = [1.0, 3.0, 2.0, 5.0, 4.0]
        result_odd = metrics._aggregate_value(values_odd, MetricAggregationType.MEDIAN)
        assert result_odd == 3.0

        # Even number of values
        values_even = [1.0, 2.0, 3.0, 4.0]
        result_even = metrics._aggregate_value(values_even, MetricAggregationType.MEDIAN)
        assert result_even == 2.5

    def test_get_aggregated_value_raw(self):
        metrics = Metrics()
        metrics.add("test_metric", 1.0)
        metrics.add("test_metric", 2.0)
        metrics.add("test_metric", 3.0)

        result = metrics.get_aggregated_value("test_metric", formatted_str=False)
        assert result == 2.0

    def test_get_aggregated_value_formatted(self):
        metrics = Metrics()
        config = MetricConfig("test", MetricAggregationType.MEAN, "{:.1f}")
        metrics.add("test_metric", 1.0, config)
        metrics.add("test_metric", 2.0)

        result = metrics.get_aggregated_value("test_metric", formatted_str=True)
        assert result == "1.5"

    def test_get_aggregated_value_not_found(self):
        metrics = Metrics()

        with pytest.raises(KeyError, match="Metric nonexistent not found"):
            metrics.get_aggregated_value("nonexistent")

    def test_get_raw_value(self):
        metrics = Metrics()
        metrics.add("test_metric", 1.0)
        metrics.add("test_metric", 2.0)

        result = metrics.get_raw_value("test_metric")
        assert result == [1.0, 2.0]

    def test_get_raw_value_not_found(self):
        metrics = Metrics()

        with pytest.raises(KeyError, match="Metric nonexistent not found"):
            metrics.get_raw_value("nonexistent")

    def test_setitem_magic_method(self):
        metrics = Metrics()
        metrics["test_metric"] = 5.0

        assert metrics.data["test_metric"] == [5.0]

    def test_getitem_magic_method(self):
        metrics = Metrics()
        metrics.add("test_metric", 1.0)
        metrics.add("test_metric", 2.0)

        result = metrics["test_metric"]
        assert result == [1.0, 2.0]

    def test_contains_magic_method(self):
        metrics = Metrics()
        metrics.add("existing_metric", 1.0)

        assert "existing_metric" in metrics
        assert "nonexistent_metric" not in metrics

    def test_len_magic_method(self):
        metrics = Metrics()
        assert len(metrics) == 0

        metrics.add("metric1", 1.0)
        assert len(metrics) == 1

        metrics.add("metric2", 2.0)
        assert len(metrics) == 2

        metrics.add("metric1", 3.0)
        assert len(metrics) == 2

    def test_hierarchical_metric_names(self):
        metrics = Metrics()

        metrics.add("actor/loss", 0.5)
        metrics.add("critic/loss", 0.3)
        metrics.add("actor/lr", 1e-4)

        assert metrics.data["actor/loss"] == [0.5]
        assert metrics.data["critic/loss"] == [0.3]
        assert metrics.data["actor/lr"] == [1e-4]

        # Check that lr config is inferred correctly
        assert metrics.configs["actor/lr"].format_str == "{:.6f}"

    def test_edge_case_empty_values(self):
        metrics = Metrics()

        metrics.add("single", 42.0)
        assert metrics.get_aggregated_value("single") == 42.0

        config_max = MetricConfig("single_max", MetricAggregationType.MAX)
        metrics.add("single_max", 43.0, config_max)
        assert metrics.get_aggregated_value("single_max") == 43.0

    def test_large_values(self):
        metrics = Metrics()

        large_values = [1e10, 2e10, 3e10]
        for val in large_values:
            metrics.add("large_metric", val)

        result = metrics.get_aggregated_value("large_metric")
        assert abs(float(result) - 2e10) < 1e8  # Mean should be 2e10

    def test_negative_values(self):
        metrics = Metrics()

        negative_values = [-5.0, -2.0, -8.0, -1.0]
        for val in negative_values:
            metrics.add("negative_metric", val)

        # Test different aggregations
        assert metrics.get_aggregated_value("negative_metric") == -4.0  # Mean

        config_max = MetricConfig("neg_max", MetricAggregationType.MAX)
        metrics.add("neg_max", -1.0, config_max)
        metrics.add("neg_max", -5.0)
        assert metrics.get_aggregated_value("neg_max") == -1.0

    def test_zero_values(self):
        metrics = Metrics()

        metrics.add("zero_metric", 0.0)
        metrics.add("zero_metric", 0.0)
        metrics.add("zero_metric", 0.0)

        assert metrics.get_aggregated_value("zero_metric") == 0.0
        assert metrics._aggregate_value([0.0, 0.0, 0.0], MetricAggregationType.STD) == 0.0

    def test_mixed_numeric_types(self):
        metrics = Metrics()

        metrics.add("mixed", 1)
        metrics.add("mixed", 2.5)
        metrics.add("mixed", np.array(3.0))
        metrics.add("mixed", torch.tensor(4.5))

        expected_mean = (1 + 2.5 + 3.0 + 4.5) / 4
        result = metrics.get_aggregated_value("mixed")
        assert abs(float(result) - expected_mean) < 1e-10

    def test_config_persistence_across_operations(self):
        config = MetricConfig("test", MetricAggregationType.MAX, "{:.1f}")
        metrics1 = Metrics()
        metrics1.add("test", 1.0, config)

        metrics2 = Metrics()
        metrics2.add("other", 2.0)

        metrics1.merge(metrics2)
        assert metrics1.configs["test"] == config

        metrics3 = Metrics()
        metrics3.add("another", 3.0)
        metrics1.update(metrics3)
        assert metrics1.configs["test"] == config

    def test_deep_copy_behavior(self):
        original_list = [1.0, 2.0, 3.0]
        data_dict = {"metric": original_list}

        metrics = Metrics.from_dict(data_dict)

        original_list.append(4.0)
        data_dict["metric"].append(5.0)

        assert metrics.data["metric"] == [1.0, 2.0, 3.0]


if __name__ == "__main__":
    unittest.main()

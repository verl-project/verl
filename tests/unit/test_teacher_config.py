import pytest
from verl.workers.config.teacher import TeacherConfig, MOPDConfig


def test_teacher_config_requires_name():
    with pytest.raises(ValueError, match="name must be non-empty"):
        TeacherConfig(name="", model_path="/models/test")


def test_teacher_config_requires_model_path():
    with pytest.raises(ValueError, match="model_path must be non-empty"):
        TeacherConfig(name="test", model_path="")


def test_mopd_config_rejects_duplicate_teacher_names():
    teachers = [
        TeacherConfig(name="math", model_path="/models/math"),
        TeacherConfig(name="math", model_path="/models/math2"),
    ]
    with pytest.raises(ValueError, match="Duplicate teacher names"):
        MOPDConfig(enabled=True, teachers=teachers)


def test_mopd_config_validates_lambda():
    with pytest.raises(ValueError, match="lambda_val must be positive"):
        MOPDConfig(enabled=False, lambda_val=0.0)


def test_mopd_config_validates_epsilon_bounds():
    with pytest.raises(ValueError, match=r"is_epsilon_low .* must be < is_epsilon_high"):
        MOPDConfig(enabled=False, is_epsilon_low=10.0, is_epsilon_high=1.0)


def test_mopd_config_rejects_empty_teachers_when_enabled():
    with pytest.raises(ValueError, match="requires at least one teacher"):
        MOPDConfig(enabled=True, teachers=[])

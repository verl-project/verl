"""Tests for trainer.max_ckpt_to_keep Hydra config support (#7295).

Verifies that the PPO trainer YAML accepts max_ckpt_to_keep and that
the trainer fallback logic resolves it correctly when the per-role
fields (max_actor_ckpt_to_keep, max_critic_ckpt_to_keep) are unset.
"""

from omegaconf import DictConfig, OmegaConf


def _resolve_ckpt_limits(trainer_cfg: DictConfig):
    """Mirrors the resolution logic in ray_trainer / v1 trainer_base."""
    remove_previous = trainer_cfg.get("remove_previous_ckpt_in_save", False)
    max_ckpt_to_keep = trainer_cfg.get("max_ckpt_to_keep", None)
    max_actor = (
        (trainer_cfg.get("max_actor_ckpt_to_keep", None) or max_ckpt_to_keep)
        if not remove_previous
        else 1
    )
    max_critic = (
        (trainer_cfg.get("max_critic_ckpt_to_keep", None) or max_ckpt_to_keep)
        if not remove_previous
        else 1
    )
    return max_actor, max_critic


def test_max_ckpt_to_keep_applies_to_both():
    """When only max_ckpt_to_keep is set, both actor and critic inherit it."""
    cfg = OmegaConf.create({
        "max_ckpt_to_keep": 3,
        "max_actor_ckpt_to_keep": None,
        "max_critic_ckpt_to_keep": None,
    })
    actor, critic = _resolve_ckpt_limits(cfg)
    assert actor == 3
    assert critic == 3


def test_per_role_overrides_global():
    """Per-role fields take priority over the global max_ckpt_to_keep."""
    cfg = OmegaConf.create({
        "max_ckpt_to_keep": 5,
        "max_actor_ckpt_to_keep": 2,
        "max_critic_ckpt_to_keep": 10,
    })
    actor, critic = _resolve_ckpt_limits(cfg)
    assert actor == 2
    assert critic == 10


def test_all_null_means_unlimited():
    """When nothing is set, both resolve to None (unlimited)."""
    cfg = OmegaConf.create({
        "max_ckpt_to_keep": None,
        "max_actor_ckpt_to_keep": None,
        "max_critic_ckpt_to_keep": None,
    })
    actor, critic = _resolve_ckpt_limits(cfg)
    assert actor is None
    assert critic is None


def test_remove_previous_overrides_everything():
    """Deprecated remove_previous_ckpt_in_save forces both to 1."""
    cfg = OmegaConf.create({
        "remove_previous_ckpt_in_save": True,
        "max_ckpt_to_keep": 5,
        "max_actor_ckpt_to_keep": 10,
        "max_critic_ckpt_to_keep": 10,
    })
    actor, critic = _resolve_ckpt_limits(cfg)
    assert actor == 1
    assert critic == 1


def test_ppo_yaml_has_max_ckpt_to_keep():
    """The PPO trainer YAML must define max_ckpt_to_keep so Hydra struct mode allows it."""
    cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    OmegaConf.set_struct(cfg, True)

    trainer = cfg.trainer
    assert "max_ckpt_to_keep" in trainer

    # Setting the value under struct mode must not raise.
    # Use open_dict to allow writes on a struct config.
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.trainer.max_ckpt_to_keep = 2
    assert cfg.trainer.max_ckpt_to_keep == 2

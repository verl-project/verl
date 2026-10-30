from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from pytorch_fob.engine.utils import log_warn, some
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter


@dataclass
class ParameterGroup:
    named_parameters: dict[str, Parameter]
    lr_multiplier: Optional[float] = field(default=None)
    weight_decay_multiplier: Optional[float] = field(default=None)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def __and__(self, other) -> "ParameterGroup":
        assert isinstance(other, ParameterGroup)
        n1 = set(self.named_parameters.keys())
        n2 = set(other.named_parameters.keys())
        all_params = self.named_parameters | other.named_parameters
        n12 = n1 & n2
        new_params = {n: all_params[n] for n in n12}
        return ParameterGroup(
            named_parameters=new_params,
            lr_multiplier=some(other.lr_multiplier, default=self.lr_multiplier),
            weight_decay_multiplier=some(
                other.weight_decay_multiplier, default=self.weight_decay_multiplier
            ),
            optimizer_kwargs=self.optimizer_kwargs | other.optimizer_kwargs,
        )

    def __len__(self) -> int:
        return len(self.named_parameters)

    def __bool__(self) -> bool:
        return not self.empty()

    def empty(self) -> bool:
        return len(self.named_parameters) == 0

    def to_optimizer_dict(
        self, lr: Optional[float] = None, weight_decay: Optional[float] = None
    ) -> dict[str, list[Parameter] | Any]:
        names = sorted(self.named_parameters)
        d = {
            "params": [self.named_parameters[n] for n in names],
            "names": names,
            **self.optimizer_kwargs,
        }
        if lr is not None:
            d["lr"] = self.lr_multiplier * lr if self.lr_multiplier is not None else lr
        if weight_decay is not None:
            d["weight_decay"] = (
                self.weight_decay_multiplier * weight_decay
                if self.weight_decay_multiplier is not None
                else weight_decay
            )
        return d


class GroupedModel(Module):
    """
    Wrapper around a nn.Module to allow specifying different optimizer settings for different parameters.
    To use this feature for your task, inherit from this class and override the `parameter_groups` method.
    Then simply wrap your model before passing it to the `__init__` method of the `TaskModel` superclass.
    """

    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def parameter_groups(self) -> list[ParameterGroup]:
        return wd_group_named_parameters(self.model)

    def grouped_parameters(
        self, lr: Optional[float] = None, weight_decay: Optional[float] = None
    ) -> list[dict[str, list[Parameter] | Any]]:
        return [
            pg.to_optimizer_dict(lr, weight_decay) for pg in self.parameter_groups()
        ]


def merge_parameter_splits(
    split1: list[ParameterGroup], split2: list[ParameterGroup]
) -> list[ParameterGroup]:
    """
    Merge two lists of ParameterGroup objects into a single list.
    Assumes that both input lists partition the parameters.
    """
    groups = []
    for pg1 in split1:
        for pg2 in split2:
            pg12 = pg1 & pg2
            if not pg12.empty():
                groups.append(pg12)
    return groups


def group_named_parameters(
    model: Module,
    g1_conds: Iterable[Callable] = (lambda *_: True,),
    g2_conds: Iterable[Callable] = (lambda *_: True,),
    special_conds: Iterable[Callable] = tuple(),
    ignore_conds: Iterable[Callable] = tuple(),
    g1_kwargs: Optional[dict[str, Any]] = None,
    g2_kwargs: Optional[dict[str, Any]] = None,
    debug: bool = False,
) -> list[ParameterGroup]:
    """
    Group named parameters based on specified conditions and return a list of ParameterGroup objects.

    Args:
        model (Module): The neural network model.
        g1_conds (Iterable[Callable]): Conditions for selecting parameters for group 1.
        g2_conds (Iterable[Callable]): Conditions for selecting parameters for group 2.
        special_conds (Iterable[Callable]): Conditions for selecting special parameters that should not be grouped.
        ignore_conds (Iterable[Callable]): Conditions for ignoring parameters (e.g. if they occur in submodules).
        g1_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for constructor of group 1.
        g2_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for constructor of group 2.

    Returns:
        List[ParameterGroup]: A list of ParameterGroup objects containing named parameters.
    """
    g1_kwargs = g1_kwargs if g1_kwargs is not None else {}
    g2_kwargs = g2_kwargs if g2_kwargs is not None else {}
    s1 = set()
    s2 = set()
    special = set()
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            elif any(c(m, p, fpn) for c in ignore_conds):
                continue
            elif any(c(m, p, fpn) for c in special_conds):
                special.add(fpn)
            elif any(c(m, p, fpn) for c in g1_conds):
                s1.add(fpn)
            elif any(c(m, p, fpn) for c in g2_conds):
                s2.add(fpn)
            elif debug:
                log_warn(
                    "group_named_parameters: Not using any rule for ",
                    fpn,
                    " in ",
                    type(m),
                )

    s1 |= param_dict.keys() - s2 - special

    # validate that we considered every parameter
    inter_params = s1 & s2
    union_params = s1 | s2
    assert (
        len(inter_params) == 0
    ), f"Parameters {str(inter_params)} made it into both s1/s2 sets!"
    assert (
        len(param_dict.keys() - special - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} \
                were not separated into either s1/s2 set!"

    if not s2:
        param_groups = [
            ParameterGroup(
                named_parameters=dict(
                    zip(
                        sorted(union_params),
                        (param_dict[pn] for pn in sorted(union_params)),
                    )
                )
            )
        ]
    else:
        param_groups = [
            ParameterGroup(
                named_parameters=dict(
                    zip(sorted(s1), (param_dict[pn] for pn in sorted(s1)))
                ),
                **g1_kwargs,
            ),
            ParameterGroup(
                named_parameters=dict(
                    zip(sorted(s2), (param_dict[pn] for pn in sorted(s2)))
                ),
                **g2_kwargs,
            ),
        ]

    return param_groups


def wd_group_named_parameters(model: Module) -> list[ParameterGroup]:
    whitelist_weight_modules = (
        nn.Linear,
        nn.modules.conv._ConvNd,
    )  # pylint: disable=protected-access # noqa
    blacklist_weight_modules = (
        nn.modules.batchnorm._NormBase,  # pylint: disable=protected-access # noqa
        nn.GroupNorm,
        nn.LayerNorm,
        nn.LocalResponseNorm,
        nn.Embedding,
    )
    ignore_modules = (nn.Sequential,)
    apply_decay_conds = [
        lambda m, _, pn: pn.endswith("weight")
        and isinstance(m, whitelist_weight_modules)
    ]
    apply_no_decay_conds = [
        lambda m, _, pn: pn.endswith("bias") or isinstance(m, blacklist_weight_modules)
    ]
    special_conds = [lambda m, p, pn: hasattr(p, "_optim")]
    ignore_conds = [lambda m, p, pn: isinstance(m, ignore_modules)]

    return group_named_parameters(
        model,
        g1_conds=apply_decay_conds,
        g2_conds=apply_no_decay_conds,
        special_conds=special_conds,
        ignore_conds=ignore_conds,
        g2_kwargs={"weight_decay_multiplier": 0.0},
    )


def resolve_parameter_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any]
) -> list[dict[str, Any]]:
    p1, p2 = dict1["params"], dict2["params"]
    n1, n2 = set(dict1["names"]), set(dict2["names"])
    n_to_p1 = dict(zip(dict1["names"], dict1["params"]))
    n_to_p2 = dict(zip(dict2["names"], dict2["params"]))
    assert len(n1) == len(p1)
    assert len(n2) == len(p2)
    kwarg1 = {k: v for k, v in dict1.items() if k not in ["params", "names"]}
    kwarg2 = {k: v for k, v in dict2.items() if k not in ["params", "names"]}
    n1_and_n2 = n1 & n2
    n1_no_n2 = n1 - n2
    n2_no_n1 = n2 - n1
    assert n1_and_n2 | n1_no_n2 | n2_no_n1 == n1 | n2
    outdict1 = {
        "params": [n_to_p1[n] for n in sorted(n1_no_n2)],
        "names": sorted(n1_no_n2),
        **kwarg1,
    }
    outdict2 = {
        "params": [n_to_p2[n] for n in sorted(n2_no_n1)],
        "names": sorted(n2_no_n1),
        **kwarg2,
    }
    # kwarg2 takes precedence if an arg is present in both dicts:
    outdict12 = {
        "params": [{**n_to_p1, **n_to_p2}[n] for n in sorted(n1_and_n2)],
        "names": sorted(n1_and_n2),
        **kwarg1,
        **kwarg2,
    }
    return [outdict1, outdict2, outdict12]


def intersect_parameter_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any]
) -> Optional[dict[str, Any]]:
    d = resolve_parameter_dicts(dict1, dict2)[2]
    return d if len(d["params"]) > 0 else None


def merge_parameter_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any]
) -> list[dict[str, Any]]:
    d = resolve_parameter_dicts(dict1, dict2)
    return list(filter(lambda x: len(x["params"]) > 0, d))

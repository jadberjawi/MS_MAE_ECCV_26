"""Small helper around hydra.utils.instantiate.

Each instantiable config carries a `name:` field for run-naming/interpolation,
which isn't a kwarg of the target class. This helper instantiates the config
after dropping any metadata keys you don't want forwarded to the constructor.
"""
from __future__ import annotations

from typing import Any, Iterable

import hydra
from omegaconf import DictConfig, OmegaConf


def instantiate_drop(
    cfg: DictConfig,
    drop_keys: Iterable[str] = ("name",),
    _recursive_: bool = False,
    **overrides,
) -> Any:
    """Instantiate a Hydra config, popping `drop_keys` first.

    Defaults to non-recursive instantiation so that strategies/heads can build
    their nested components themselves (passing the shared encoder, etc.).
    """
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for k in drop_keys:
        if k in cfg:
            del cfg[k]
    return hydra.utils.instantiate(cfg, _recursive_=_recursive_, **overrides)

"""YAML config loading: experiment files are deep-merged over configs/default.yaml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access; nested dicts become Configs."""

    def __init__(self, data=None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            if k not in out:
                raise KeyError(f"unknown config key: {k}")
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    with open(DEFAULT_CONFIG) as f:
        data = yaml.safe_load(f)
    if path is not None:
        with open(path) as f:
            data = deep_merge(data, yaml.safe_load(f) or {})
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data)

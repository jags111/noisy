from __future__ import annotations
from typing import Dict, Any, Optional
import os
from logging import getLogger
from pathlib import Path
import yaml


logger = getLogger('noisy.utils')


class AttrDict(Dict[str, Any]):
    """A dictionary with JavaScript-like syntax. I.e. instead of d["my_key"],
    we can simply say d.my_key."""

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key not in self:
                raise e
            return self[key]

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AttrDict:
        return AttrDict(**{k: cls._from_obj(v) for k, v in d.items()})

    @classmethod
    def _from_obj(cls, o: Any) -> Any:
        if isinstance(o, dict):
            return cls.from_dict(o)
        if isinstance(o, list):
            return [cls._from_obj(x) for x in o]
        return o

    def to_dict(self) -> Dict[str, Any]:

        def _to_obj(x: Any) -> Any:
            if isinstance(x, AttrDict):
                return {k: _to_obj(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_obj(i) for i in x]
            return x

        obj = _to_obj(self)
        assert isinstance(obj, dict)
        return obj

    @classmethod
    def from_yaml(cls, path: Path) -> AttrDict:
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d)

    def to_yaml(self, path: Path) -> None:
        with open(path, 'w') as fh:
            yaml.dump(self.to_dict(), fh)


def make_symlink(sl: Path, cp: Path) -> None:
    if sl.exists():
        sl.unlink()
    sl.symlink_to(rel_path(cp.absolute(), sl.absolute().parent))


def rel_path(path: Path, root: Optional[Path] = None) -> Path:
    if root is None:
        root = Path(os.getcwd())
    return Path(os.path.relpath(path, root))

import enum
import os
from typing import Any

from loguru import logger


TRUST_REMOTE_CODE = os.environ.get("OM_TRUST_REMOTE_CODE", "true").lower() in (
    "true",
    "1",
    "t",
    "on",
)


class FrameWork(enum.Enum):
    PT = enum.auto()
    MS = enum.auto()


class Registry:
    """
    A utility class to register and retrieve values by keys.
    """

    def __init__(self):
        self._map = {}

    def register(self, keys: Any, value: Any):
        try:
            _ = iter(keys)
            is_iterable = True
        except TypeError:
            is_iterable = False

        if isinstance(keys, str) or not is_iterable:
            keys = [keys]

        for key in keys:
            if key in self._map:
                logger.warning(
                    f'Overriding previously registered "{key}" value "{value}"'
                )
            self._map[key] = value

    def get(self, key: Any) -> Any:
        if key in self._map:
            return self._map[key]
        return None

    def keys(self):
        return self._map.keys()

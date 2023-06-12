import json
import hashlib
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import os

import diskcache
import platformdirs


class Cache(ABC):
    @abstractmethod
    def __getitem__(self, key: str) -> str:
        """get an item from the cache or throw key error"""
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: str) -> None:
        """set an item in the cache"""
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """see if we can return a cached value for the passed key"""
        pass

    def create_key(self, llm: str, **kwargs: Dict[str, Any]) -> str:
        """Define a lookup key for a call to the given llm with the given kwargs.
        One of the keyword args could be `cache_key` in which case this function should respect that
        and use it.
        """
        if "cache_key" in kwargs:
            return str(kwargs["cache_key"])

        hasher = hashlib.md5()
        options_str = json.dumps(kwargs, sort_keys=True)

        combined = "{}{}".format(llm, options_str).encode()

        hasher.update(combined)
        return hasher.hexdigest()
    
    def clear(self):
        raise NotImplementedError()





class DiskCache(Cache):
    """DiskCache is a cache that uses diskcache lib."""
    def __init__(self, llm_name: str):
        self._diskcache = diskcache.Cache(
            os.path.join(
                platformdirs.user_cache_dir("guidance"), f"_{llm_name}.diskcache"
            )
        )

    def __getitem__(self, key: str) -> str:
        return self._diskcache[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._diskcache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._diskcache
    
    def clear(self):
        self._diskcache.clear()



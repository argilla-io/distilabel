# Copyright 2023-present, Argilla, Inc.
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

"""
`datasketch` (https://github.com/ekzhu/datasketch) doesn't offer a way to store the hash tables in disk. This
is a custom implementation that uses `diskcache` to store the hash tables in disk.
Note: This implementation is not optimized for performance, but could be worth
creating a PR to `datasketch`.
"""

import shutil
import struct
from pathlib import Path
from typing import Callable, Dict, Final, Optional, Tuple

from datasketch import MinHashLSH as _MinHashLSH
from datasketch.lsh import _optimal_param
from datasketch.storage import OrderedStorage, UnorderedStorage, _random_name
from datasketch.storage import ordered_storage as _ordered_storage
from datasketch.storage import unordered_storage as _unordered_storage

KEY_VALUE_DISK_DIR: Path = Path.home() / ".cache" / "distilabel" / "key_value_store"
KV_DISK_LIST_NAME: Final[str] = "disckache_list_storage"
KV_DISK_SET_NAME: Final[str] = "diskcache_set_storage"


class DiskCacheListStorage(OrderedStorage):
    def __init__(self, config, name) -> None:
        path = config.get("path", self._get_db_name(name))
        try:
            from diskcache import Index
        except ImportError as e:
            raise ImportError(
                "`diskcache` is required for disk storage using `MinHashDedup`. "
                "Please install it using `pip install diskcache`."
            ) from e

        # Start with a clean file on each pipeline
        if Path(path).exists():
            shutil.rmtree(path)
        self._db = Index(path)

    def _get_db_name(self, name):
        return str(KEY_VALUE_DISK_DIR / f"{name}_{KV_DISK_LIST_NAME}")

    def keys(self):
        return self._db.keys()

    def get(self, key):
        return self._db.get(key, [])

    def remove(self, *keys):
        self._db.clear()

    def remove_val(self, key, val):
        self.get(key).remove(val)

    def insert(self, key, *vals, **kwargs):
        res = self.get(key)
        res.extend(vals)
        self._db[key] = res

    def size(self):
        return len(self._db)

    def itemcounts(self, **kwargs):
        return {k: len(v) for k, v in self._db.items()}

    def has_key(self, key):
        return key in self._db

    def close(self):
        self._db._cache.close()


class DiskCacheSetStorage(UnorderedStorage, DiskCacheListStorage):
    def _get_db_name(self, name):
        return str(KEY_VALUE_DISK_DIR / f"{name}_{KV_DISK_SET_NAME}")

    def get(self, key):
        return self._db.get(key, set())

    def insert(self, key, *vals, **kwargs):
        res = self.get(key)
        res.update(vals)
        self._db[key] = res


def ordered_storage(config, name=None):
    """Copy of `datasketch.storage.ordered_storage` with the addition of `DiskCacheListStorage`."""
    tp = config["type"]
    if tp == "disk":
        return DiskCacheListStorage(config, name=name)
    return _ordered_storage(config, name=name)


def unordered_storage(config, name=None):
    """Copy of `datasketch.storage.ordered_storage` with the addition of `DiskCacheSetStorage`."""
    tp = config["type"]
    if tp == "disk":
        return DiskCacheSetStorage(config, name=name)
    return _unordered_storage(config, name=name)


class MinHashLSH(_MinHashLSH):
    """Custom implementation of `datasketch.MinHashLSH` to allow passing a custom
    storage configuration to store the hash tables in disk.

    This could be merged in the original repository, the only changes
    to the __init__ are the additional `close` method, and the use
    of our custom `ordered_storage` and `unordered_storage` functions.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        params: Optional[Tuple[int, int]] = None,
        storage_config: Optional[Dict] = None,
        prepickle: Optional[bool] = None,
        hashfunc: Optional[Callable[[bytes], bytes]] = None,
    ) -> None:
        storage_config = {"type": "dict"} if not storage_config else storage_config
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    "The product of b and r in params is "
                    "{} * {} = {} -- it must be less than num_perm {}. "
                    "Did you forget to specify num_perm?".format(
                        self.b, self.r, self.b * self.r, num_perm
                    )
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(
                threshold, num_perm, false_positive_weight, false_negative_weight
            )
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")

        self.prepickle = (
            storage_config["type"] == "redis" if not prepickle else prepickle
        )

        self.hashfunc = hashfunc
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        basename = storage_config.get("basename", _random_name(11))
        self.hashtables = [
            unordered_storage(
                storage_config,
                name=b"".join([basename, b"_bucket_", struct.pack(">H", i)]),
            )
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.keys = ordered_storage(storage_config, name=b"".join([basename, b"_keys"]))

    def close(self):
        """Closes the internal connections."""
        if isinstance(self.hashtables[0], DiskCacheListStorage):
            for ht in self.hashtables:
                ht.close()
            self.keys.close()

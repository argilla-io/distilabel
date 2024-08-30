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
`dataskech` (https://github.com/ekzhu/datasketch) doesn't offer a way to store the hash tables in disk. This
is a custom implementation that uses `shelve` to store the hash tables in disk.
Note: This implementation is not optimized for performance, but could be worth
creating a PR to `datasketch`.
"""

import shelve
import shutil
import struct
from pathlib import Path
from typing import Callable, Dict, Final, Optional, Tuple

from datasketch import MinHashLSH as _MinHashLSH
from datasketch.lsh import _optimal_param
from datasketch.storage import OrderedStorage, UnorderedStorage, _random_name
from datasketch.storage import ordered_storage as _ordered_storage
from datasketch.storage import unordered_storage as _unordered_storage

SHELVE_DIR: Path = Path.home() / ".cache" / "distilabel" / "key_value_store"
SHELVE_LIST_NAME: Final[str] = "shelve_list_storage"
SHELVE_SET_NAME: Final[str] = "shelve_set_storage"

KEY_VALUE_DISK_DIR: Path = Path.home() / ".cache" / "distilabel" / "key_value_store"
KV_DISK_LIST_NAME: Final[str] = "disckache_list_storage"
KV_DISK_SET_NAME: Final[str] = "diskcache_set_storage"


def _custom_shelve_open(path, writeback=True):
    from shelve import Shelf

    class CustomShelve(Shelf):
        def __init__(self, filename, flag="c", protocol=None, writeback=False):
            import dbm.gnu as dbm

            Shelf.__init__(self, dbm.open(filename, flag), protocol, writeback)

    return CustomShelve(path, flag="n", protocol=None, writeback=writeback)


class ShelveListStorage(OrderedStorage):
    """Key/Value storage using shelve to store the hash tables in disk.
    It mimics the behaviour of `datasketch.DictListStorage`.
    The only difference is the storage in disk.
    The functionality is on purpose to avoid unnecessary errors.
    """

    def __init__(self, config, name) -> None:
        path = config.get("path", self._get_db_name(name))
        # Read about writeback here: https://docs.python.org/3/library/shelve.html#shelve.open
        writeback = config.get("writeback", True)
        # The flag is set to "n" to recreate the file always, we assume
        # every pipeline works on it's own and recomputes it instead of trusting
        # the cache.
        # Note: Maybe we could move this to use `diskcache` instead of shelve
        self._db = shelve.open(path, writeback=writeback, flag="n")
        # self._db = _custom_shelve_open(path, writeback=writeback)

    def _get_db_name(self, name):
        return str(SHELVE_DIR / f"{name}_{SHELVE_LIST_NAME}")

    def keys(self):
        return self._db.keys()

    def get(self, key):
        return self._db.get(str(key), [])

    def remove(self, *keys):
        for key in keys:
            del self._db[str(key)]

    def remove_val(self, key, val):
        self._db[str(key)].remove(val)

    def insert(self, key, *vals, **kwargs):
        key = str(key)
        if not self._db.get(key):
            self._db[key] = []
        self._db[key].extend(vals)

    def size(self):
        return len(self._db)

    def itemcounts(self, **kwargs):
        return {k: len(v) for k, v in self._db.items()}

    def has_key(self, key):
        return key in self._db

    def close(self):
        self._db.close()


class ShelveSetStorage(UnorderedStorage, ShelveListStorage):
    """Key/Value storage using shelve to store the hash tables in disk.
    It mimics the behaviour of `datasketch.DictSetStorage`.
    The only difference is the storage in disk.
    The functionality is on purpose to avoid unnecessary errors.
    """

    def _get_db_name(self, name):
        return str(SHELVE_DIR / f"{name}_{SHELVE_SET_NAME}")

    def get(self, key):
        return self._db.get(str(key), set())

    def insert(self, key, *vals, **kwargs):
        key = str(key)
        if not self._db.get(key):
            self._db[key] = set()
        self._db[key].update(vals)


class DiskCacheListStorage(OrderedStorage):
    def __init__(self, config, name) -> None:
        path = config.get("path", self._get_db_name(name))
        # Read about writeback here: https://docs.python.org/3/library/shelve.html#shelve.open
        # The flag is set to "n" to recreate the file always, we assume
        # every pipeline works on it's own and recomputes it instead of trusting
        # the cache.
        # Note: Maybe we could move this to use `diskcache` instead of shelve
        # self._db = shelve.open(path, writeback=writeback, flag="n")
        # self._db = _custom_shelve_open(path, writeback=writeback)

        # from diskcache import Cache as KeyValStore
        # self._db = KeyValStore(path)
        from diskcache import Index

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
        self._db.close()


class DiskCacheSetStorage(UnorderedStorage, DiskCacheListStorage):
    def _get_db_name(self, name):
        return str(KEY_VALUE_DISK_DIR / f"{name}_{SHELVE_SET_NAME}")

    def get(self, key):
        return self._db.get(key, set())

    def insert(self, key, *vals, **kwargs):
        res = self.get(key)
        res.update(vals)
        self._db[key] = res


def ordered_storage(config, name=None):
    """Copy of `datasketch.storage.ordered_storage` with the addition of `ShelveListStorage`."""
    tp = config["type"]
    if tp == "disk":
        # return ShelveListStorage(config, name=name)
        return DiskCacheListStorage(config, name=name)
    return _ordered_storage(config, name=name)


def unordered_storage(config, name=None):
    """Copy of `datasketch.storage.ordered_storage` with the addition of `ShelveSetStorage`."""
    tp = config["type"]
    if tp == "disk":
        # return ShelveSetStorage(config, name=name)
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
        """Closes the shelve objects."""
        if isinstance(self.hashtables[0], ShelveListStorage):
            for ht in self.hashtables:
                ht.close()
            self.keys.close()

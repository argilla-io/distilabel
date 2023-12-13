from concurrent.futures import Future
from typing import List, TypeGuard, TypeVar, Union

T = TypeVar("FutureResult")


def is_list_of_futures(
    results: Union[List[Future[T]], List[List[T]]],
) -> TypeGuard[List[Future[T]]]:
    """Check if results is a list of futures. This function narrows the type of
    `results` to `List[Future[T]]` if it is a list of futures.

    Args:
        results: A list of futures.

    Returns:
        `True` if `results` is a list of futures, `False` otherwise.
    """
    return isinstance(results, list) and isinstance(results[0], Future)

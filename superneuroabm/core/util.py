import time
from typing import Iterable, List, Any, Tuple, Optional

import numpy as np


def convert_to_equal_side_tensor(
    tensor: List[Any], max_dims: Optional[Tuple[int]] = None
) -> np.array:
    dim2maxlen = {}

    def find_max_depth(l, curr_depth=0):
        dim2maxlen[curr_depth] = max(dim2maxlen.get(curr_depth, 0), len(l))
        max_depth = curr_depth
        for item in l:
            if type(item) == list:
                max_depth = max(
                    max_depth, find_max_depth(item, curr_depth + 1)
                )
        return max_depth

    start = time.time()
    if not max_dims:
        print("no dims specified... running find max depth")
        find_max_depth(tensor)
        max_dims = tuple(list(dim2maxlen.values()))
    answer = np.full(shape=max_dims, fill_value=np.nan)

    def fill_arr(arr, coord):
        if len(coord) == len(max_dims):
            if type(arr) == list:
                raise TypeError()
            answer[tuple(coord)] = arr
        else:
            if type(arr) != list:
                raise TypeError()
            for i, item in enumerate(arr):
                new_coord = coord + [i]
                fill_arr(item, new_coord)

    start = time.time()
    fill_arr(tensor, [])
    return answer


def compress_tensor(arr: Iterable, level: int = 0):
    if not hasattr(arr, "__iter__") and not hasattr(
        arr, "__cuda_array_interface__"
    ):
        if not np.isnan(arr):
            return arr
        else:
            return None
    else:
        new_arr = []
        for item in arr:
            new_item = compress_tensor(item, level + 1)
            if (
                (not isinstance(new_item, Iterable) and new_item != None)
                or (isinstance(new_item, Iterable) and len(new_item))
                or level <= 0
            ):
                new_arr.append(new_item)
        if len(new_arr):
            return new_arr
        else:
            return [] if level else None

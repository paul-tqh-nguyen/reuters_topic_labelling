#!/usr/bin/python3 -OO

"""
This file contains miscellaneous utilities (mostly for debugging).

Sections:
* Imports
* Misc. Utilities
"""

###########
# Imports #
###########

import os
import sys
import traceback
import pdb
import inspect
import time
from collections import Counter
from tqdm import tqdm
from contextlib import contextmanager
from typing import Iterable, List, Callable

################### 
# Misc. Utilities #
###################

def p1(iterable: Iterable) -> None:
    for e in iterable:
        print(e)
    return

def current_tensors() -> List:
    import torch
    import gc
    return [e for e in gc.get_objects() if isinstance(e, torch.Tensor)]

def only_one(items: List):
    assert isinstance(items, list)
    assert len(items) == 1
    return items[0]

def at_most_one(items: List):
    return only_one(items) if items else None

def parallel_map(func: Callable, iterable: Iterable) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.map(func, iterable)
    p.close()
    p.join()
    return result

def eager_map(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

def eager_filter(func: Callable, iterable: Iterable) -> List:
    return list(filter(func, iterable))

def implies(antecedent: bool, consequent: bool) -> bool:
    return not antecedent or consequent

def histogram(iterator: Iterable) -> Counter:
    counter = Counter()
    for element in iterator:
        counter[element]+=1
    return counter

@contextmanager
def safe_cuda_memory():
    try:
        yield
    except RuntimeError as err:
        if 'CUDA out of memory' not in str(err):
            raise
        else:
            print("CUDA ran out of memory.")

@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[], None] = None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print(f'{section_name.strip()} took {elapsed_time} seconds.')
    else:
        print(f'Execution took {elapsed_time} seconds.')

def _dummy_tqdm_message_func(index: int):
    return ''

def tqdm_with_message(iterable,
                      pre_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      post_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      *args, **kwargs):
    progress_bar_iterator = tqdm(iterable, *args, **kwargs)
    for index, element in enumerate(progress_bar_iterator):
        if pre_yield_message_func != _dummy_tqdm_message_func:
            pre_yield_message = pre_yield_message_func(index)
            progress_bar_iterator.set_description(pre_yield_message)
            progress_bar_iterator.refresh()
        yield element
        if post_yield_message_func != _dummy_tqdm_message_func:
            post_yield_message = post_yield_message_func(index)
            progress_bar_iterator.set_description(post_yield_message)
            progress_bar_iterator.refresh()

def debug_on_error(func: Callable) -> Callable:
    def decorating_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return decorating_function

if __name__ == '__main__':
    print("This file contains miscellaneous utilities.")

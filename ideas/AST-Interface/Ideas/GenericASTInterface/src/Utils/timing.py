from functools import wraps
from time import time
from typing import final

def measure(func):
  @wraps(func)
  def time_func(*args, **kwargs):
    start = int(round(time() * 1000))

    try:
      return func(*args, **kwargs)
    finally:
      end = int(round(time() * 1000))

    print("Func execution with name: {} took {} ms".format(func.__name__, end - start))

  return time_func

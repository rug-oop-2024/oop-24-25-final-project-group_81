from functools import wraps
from typing import Any, Callable
import streamlit as st

def title(title: str) -> Callable[...,Any]:
    """
    Wraps a function and prints a title before execution.

    :param function: the wrapped function
    :type function: Callable[...,Any]
    :param title: the title to be printed
    :type title: str
    :return: whathever the wrapped function does.
    :rtype: Callable[...,Any]
    """
    def decorator(function: Callable[...,Any]) -> Callable[...,Any]:
        """
        The decorator.
        """
        @wraps(function)
        def wrapped_function(*args: Any, **kwargs: Any) -> Any:
            st.write(f"# {title}")
            result = function(*args, **kwargs)
            return result
        return wrapped_function
    return decorator

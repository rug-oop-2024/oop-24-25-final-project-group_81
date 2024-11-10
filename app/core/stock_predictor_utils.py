from functools import wraps
from typing import Callable
import streamlit as st


def title(title: str) -> Callable[..., any]:
    """
    Wraps a function and prints a title before execution.

    :param function: the wrapped function
    :type function: Callable[...,Any]
    :param title: the title to be printed
    :type title: str
    :return: whathever the wrapped function does.
    :rtype: Callable[...,Any]
    """

    def decorator(function: Callable[..., any]) -> Callable[..., any]:
        """
        The decorator.
        """

        @wraps(function)
        def wrapped_function(*args: any, **kwargs: any) -> any:
            st.write(f"# {title}")
            result = function(*args, **kwargs)
            return result

        return wrapped_function

    return decorator

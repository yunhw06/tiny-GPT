from typing import Any

class Tensor:
    ...

class device:
    ...

class dtype:
    ...

long: dtype


def no_grad() -> Any:
    ...


def zeros(*args: Any, **kwargs: Any) -> Tensor:
    ...


def tensor(*args: Any, **kwargs: Any) -> Tensor:
    ...


def multinomial(*args: Any, **kwargs: Any) -> Tensor:
    ...


def arange(*args: Any, **kwargs: Any) -> Tensor:
    ...


def cat(*args: Any, **kwargs: Any) -> Tensor:
    ...


def tril(*args: Any, **kwargs: Any) -> Tensor:
    ...

from . import nn

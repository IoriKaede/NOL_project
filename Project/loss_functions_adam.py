import numpy as np

from supplementary import Value


def mse_loss(x: Value, y: Value) -> Value:
    diff = x.data-y.data

    if len(diff.shape) == 2:  # batch case
        r = diff.shape[0]
    else:  # single data tuple case
        r = 1

    result = Value(1 / r * np.sum(diff ** 2), f"$\frac{1}{r}||{x.expr}-{y.expr}||^2$", (x, y))

    def _backward_gradient_step():
        x.grad += 2/r * diff * result.grad
        y.grad -= 2/r * diff * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result

def ce_loss(x: Value, y: Value) -> Value:
    eps = 1e-9
    diff = x.data-y.data
    if len(diff.shape) == 2:
        r = diff.shape[0]
    else:
        r = 1

    result = Value(-1 / r * np.sum(y.data * np.log(x.data + eps)), f"$\frac{1}{r}{y.expr}log({x.expr})$", (x, y))

    def _backward_gradient_step():
        x.grad += 1/r * diff * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result

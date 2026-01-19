import numpy as np

from supplementary import Value


def mse_loss(x: Value, y: Value) -> Value:
    diff = x.data-y.data

    if len(diff.shape) == 2:  # batch case
        r = diff.shape[0]
    else:  # single data tuple case
        r = 1

    result = Value(1 / r * np.sum(diff ** 2), "MSE", (x, y))

    def _backward_gradient_step():
        x.grad += 2/r * diff * result.grad
        y.grad -= 2/r * diff * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result


def cross_entropy_loss(y_pred: Value, y_true: Value) -> Value:
    epsilon = 1e-15
    y_pred_safe = np.clip(y_pred.data, epsilon, 1 - epsilon)

    if len(y_pred_safe.shape) == 2:
        N = y_pred_safe.shape[0]
    else:
        N = 1

    loss_val = -np.sum(y_true.data * np.log(y_pred_safe)) / N
    result = Value(loss_val, f"CCE_Loss", (y_pred, y_true))

    def _backward_gradient_step():
        grad = -(y_true.data / y_pred_safe) / N
        y_pred.grad += grad * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result


def mae_loss(x: Value, y: Value) -> Value:
    diff = x.data - y.data
    if len(diff.shape) == 2:  # batch case
        r = diff.shape[0]
    else:  # single data tuple case
        r = 1
    result = Value(np.abs(1 / r * np.sum(diff ** 2)), "MAE", (x, y))

    def _backward_gradient_step():
        x.grad += 2 / r * diff * result.grad
        y.grad -= 2 / r * diff * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result


def hinge_loss(x: Value, y: Value, margin=1.0) -> Value:   #code from geeks
    scores = x.data
    y_true_idx = np.argmax(y.data, axis=1)
    correct_scores = scores[np.arange(len(scores)), y_true_idx][:, np.newaxis]
    margins = np.maximum(0, scores - correct_scores + margin)
    margins[np.arange(len(scores)), y_true_idx] = 0

    loss_val = np.sum(margins) / scores.shape[0]
    result = Value(loss_val, "Hinge", (x, y))

    def _backward_gradient_step():
        grad = np.zeros_like(scores)
        grad[margins > 0] = 1
        grad[np.arange(len(scores)), y_true_idx] = -np.sum(margins > 0, axis=1)
        x.grad += (grad / scores.shape[0]) * result.grad
    result._backward_gradient_step = _backward_gradient_step
    return result
import numpy as np
from supplementary import Value

# 1. MSE Loss
def mse_loss(x: Value, y: Value) -> Value:
    diff = x.data - y.data
    if len(diff.shape) == 2:
        r = diff.shape[0]
    else:
        r = 1

    result = Value(1 / r * np.sum(diff ** 2), f"MSE", (x, y))

    def _backward_gradient_step():
        x.grad += 2/r * diff * result.grad
        y.grad -= 2/r * diff * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result

# 2. MAE Loss
def mae_loss(x: Value, y: Value) -> Value:
    diff = x.data - y.data
    if len(diff.shape) == 2:
        r = diff.shape[0]
    else:
        r = 1

    result = Value(1 / r * np.sum(np.abs(diff)), f"MAE", (x, y))

    def _backward_gradient_step():
        grad_input = (1 / r) * np.sign(diff)
        x.grad += grad_input * result.grad
        y.grad -= grad_input * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result

# 3. Cross Entropy Loss
def cross_entropy_loss(x: Value, y: Value) -> Value:
    eps = 1e-9
    if len(x.data.shape) == 2:
        r = x.data.shape[0]
    else:
        r = 1
    loss_val = -np.sum(y.data * np.log(x.data + eps)) / r
    result = Value(loss_val, f"CrossEntropy", (x, y))

    def _backward_gradient_step():
        grad_input = -(y.data / (x.data + eps))
        x.grad += (1 / r) * grad_input * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result

# 4. Hinge Loss
def hinge_loss(x: Value, y: Value, delta: float = 1.0) -> Value:
    if len(x.data.shape) == 2:
        r = x.data.shape[0]
    else:
        r = 1
    scores = x.data
    labels = y.data
    correct_class_scores = np.sum(scores * labels, axis=1, keepdims=True)

    margins = np.maximum(0, scores - correct_class_scores + delta)
    margins[labels.astype(bool)] = 0

    loss_val = np.sum(margins) / r

    result = Value(loss_val, f"Hinge", (x, y))

    def _backward_gradient_step():
        binary_mask = (margins > 0).astype(np.float64)
        row_sum = np.sum(binary_mask, axis=1, keepdims=True)
        dscores = binary_mask
        dscores[labels.astype(bool)] = -row_sum.flatten()

        dscores /= r
        x.grad += dscores * result.grad
        y.grad -= 0

    result._backward_gradient_step = _backward_gradient_step
    return result
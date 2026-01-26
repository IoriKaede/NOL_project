import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from activation_functions import relu, softmax
from data_loader import DataLoader
from models import NeuralNetwork
from models_adam import NeuralNetwork_Adam
from supplementary import Value, load_mnist
import loss_functions as lf

data_dir = Path(__file__).resolve().parent / "data"
train_images, train_y = load_mnist(data_dir, kind='train')
test_images, test_y = load_mnist(data_dir, kind='t10k')

train_images = train_images.reshape(60_000, 784) / 255
test_images = test_images.reshape(10_000, 784) / 255

train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1
test_labels = np.zeros((10_000, 10))
test_labels[np.arange(10_000), test_y] = 1

val_split = 5000
val_images, val_labels = train_images[:val_split], train_labels[:val_split]
train_images_sub, train_labels_sub = train_images[val_split:], train_labels[val_split:]

train_loader = DataLoader(list(zip(train_images_sub, train_labels_sub)), batch_size=64, shuffle=True)
val_loader = DataLoader(list(zip(val_images, val_labels)), batch_size=64, shuffle=False)
test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=64, shuffle=False)

epochs = 50
lr_sgd = 0.01
lr_adam = 0.001
loss_registry = {"MSE": lf.mse_loss, "MAE": lf.mae_loss, "CrossEntropy": lf.cross_entropy_loss, "Hinge": lf.hinge_loss}


results = {k: {
    "SGD": {"val_acc": [], "test_acc": [], "val_loss": [], "test_loss": []},
    "Adam": {"val_acc": [], "test_acc": [], "val_loss": [], "test_loss": []}
} for k in loss_registry}


def evaluate(model, loader, loss_func):
    total_loss = 0
    correct = 0
    total_samples = 0
    for batch in loader:
        x_val = Value(np.vstack([b[0] for b in batch]))
        y_val = Value(np.vstack([b[1] for b in batch]))
        out = model(x_val)
        total_loss += loss_func(out, y_val).data
        correct += np.sum(np.argmax(out.data, axis=1) == np.argmax(y_val.data, axis=1))
        total_samples += len(batch)
    return total_loss / len(loader), correct / total_samples


for loss_name, loss_func in loss_registry.items():
    for opt_name in ["SGD", "Adam"]:
        print(f"\nExperiment: {loss_name} | Optimizer: {opt_name}")

        if opt_name == "Adam":
            model = NeuralNetwork_Adam(layers=[784, 128, 64, 10], activation_functions=[relu, relu, softmax])
            lr = lr_adam
        else:
            model = NeuralNetwork(layers=[784, 128, 64, 10], activation_functions=[relu, relu, softmax])
            lr = lr_sgd

        for epoch in range(epochs):
            # Train
            for batch in tqdm(train_loader, desc=f"Ep {epoch + 1}", leave=False):
                model.reset_gradients()
                x, y = Value(np.vstack([b[0] for b in batch])), Value(np.vstack([b[1] for b in batch]))
                loss = loss_func(model(x), y)
                loss.backward()
                model.adam_step(lr) if opt_name == "Adam" else model.gradient_descent(lr)

            # Evaluate Validation Set
            v_loss, v_acc = evaluate(model, val_loader, loss_func)
            # Evaluate Test Set (Now recorded every epoch)
            t_loss, t_acc = evaluate(model, test_loader, loss_func)

            # Store everything
            results[loss_name][opt_name]["val_loss"].append(v_loss)
            results[loss_name][opt_name]["val_acc"].append(v_acc)
            results[loss_name][opt_name]["test_loss"].append(t_loss)
            results[loss_name][opt_name]["test_acc"].append(t_acc)

fig, axes = plt.subplots(4, 2, figsize=(20, 24))
fig.suptitle(f"SGD vs Adam")

for i, loss_name in enumerate(loss_registry.keys()):
    ax_acc = axes[i, 0]
    ax_loss = axes[i, 1]

    for opt, col in zip(["SGD", "Adam"], ["tab:blue", "tab:orange"]):
        res = results[loss_name][opt]
        epochs_range = range(1, epochs + 1)

        ax_acc.plot(epochs_range, res["val_acc"], label=f"{opt} Validation Acc", color=col, linestyle= '--')
        ax_acc.plot(epochs_range, res["test_acc"], label=f"{opt} Test Acc", color=col)

        ax_loss.plot(epochs_range, res["val_loss"], label=f"{opt} Validation Loss", color=col, linestyle= '--')
        ax_loss.plot(epochs_range, res["test_loss"], label=f"{opt} Test Loss", color=col)

    ax_acc.set_title(f"{loss_name}: Accuracy Comparison", fontweight='bold')
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Epochs")
    ax_acc.legend(fontsize='small', ncol=2)
    ax_acc.grid(True, alpha=0.3)

    ax_loss.set_title(f"{loss_name}: Loss Comparison", fontweight='bold')
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.legend(fontsize='small', ncol=2)
    ax_loss.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
print("\n" + "=" * 85)
print(f"{'Loss Function':<15} | {'Opt':<5} | {'Val Acc':<10} | {'Test Acc':<10} | {'Val Loss':<10} | {'Test Loss':<10}")
print("-" * 85)
for ln in loss_registry:
    for opt in ["SGD", "Adam"]:
        r = results[ln][opt]
        print(
            f"{ln:<15} | {opt:<5} | {r['val_acc'][-1]:<10.4f} | {r['test_acc'][-1]:<10.4f} | {r['val_loss'][-1]:<10.4f} | {r['test_loss'][-1]:<10.4f}")


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


from activation_functions import logi, softmax
from data_loader import DataLoader
from models import NeuralNetwork
from supplementary import Value, load_mnist

from four_loss_function import mse_loss, cross_entropy_loss


EPOCHS = 50
LEARNING_RATE = 3e-3
BATCH_SIZE = 32


print("loading data...")
data_dir = Path(__file__).resolve().parent / "data"
try:
    train_images, train_y = load_mnist(data_dir, kind='train')
    test_images, test_y = load_mnist(data_dir, kind='t10k')
except Exception as e:
    print("error")
    raise e


train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0
# One-hot
train_labels = np.zeros((len(train_y), 10));
train_labels[np.arange(len(train_y)), train_y] = 1
test_labels = np.zeros((len(test_y), 10));
test_labels[np.arange(len(test_y)), test_y] = 1

# DataLoader
train_loader = DataLoader(list(zip(train_images, train_labels)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=BATCH_SIZE, shuffle=False)
train_dataset_size = len(train_images)
test_dataset_size = len(test_images)


network_configs = {
    "Shallow (1 Hidden)": {
        "layers": [784, 64, 10],
        "activations": [logi, softmax]
    },
    "Medium (3 Hidden)": {
        "layers": [784, 256, 128, 64, 10],
        "activations": [logi, logi, logi, softmax]
    },
    "Deep (5 Hidden)": {
        "layers": [784, 256, 256, 128, 128, 64, 10],
        "activations": [logi, logi, logi, logi, logi, softmax]
    }
}

loss_functions = {
    "MSE": mse_loss,
    "Cross Entropy": cross_entropy_loss
}


results = {name: {} for name in network_configs}

print(f"\n{'=' * 60}")
print(f"testing (Sub-question-6)")
print(f"{'=' * 60}\n")

for depth_name, config in network_configs.items():
    print(f"  testing: {depth_name}")
    print(f"    Layers: {config['layers']}")

    for loss_name, loss_fn in loss_functions.items():
        print(f"    ->  Loss: {loss_name} ... ", end="")

        nn = NeuralNetwork(
            layers=config['layers'],
            activation_functions=config['activations']
        )

        for epoch in range(1, EPOCHS + 1):
            # Train
            for batch in train_loader:
                nn.reset_gradients()
                imgs = np.vstack([x[0] for x in batch])
                lbls = np.vstack([x[1] for x in batch])

                out = nn(Value(imgs, "X"))
                loss = loss_fn(out, Value(lbls, "Y"))
                loss.backward()
                nn.gradient_descent(LEARNING_RATE)

        correct = 0
        for batch in test_loader:
            imgs = np.vstack([x[0] for x in batch])
            lbls = np.vstack([x[1] for x in batch])
            out = nn(Value(imgs, "X"))

            pred = np.argmax(out.data, axis=1)
            true = np.argmax(lbls, axis=1)
            correct += np.sum(pred == true)

        final_acc = correct / test_dataset_size
        results[depth_name][loss_name] = final_acc
        print(f"Final Acc: {final_acc:.4f}")


print(f"\n{'=' * 70}")
print(f" result")
print(f"{'=' * 70}")
print(f"{'Network Depth':<20} | {'MSE Acc':<10} | {'CE Acc':<10} | {'Gap (CE - MSE)':<15}")
print("-" * 70)

gap_values = []
depth_labels = []

for depth_name in network_configs:
    mse_acc = results[depth_name]["MSE"]
    ce_acc = results[depth_name]["Cross Entropy"]
    gap = ce_acc - mse_acc

    gap_values.append(gap)
    depth_labels.append(depth_name.split()[0])  # 取 'Shallow', 'Medium' 等词

    print(f"{depth_name:<20} | {mse_acc:.4f}     | {ce_acc:.4f}     | {gap:+.4f}")

print(f"{'=' * 70}")

# === 简单的可视化 ===
plt.figure(figsize=(8, 5))
plt.bar(depth_labels, gap_values, color=['skyblue', 'orange', 'salmon'])
plt.title("Performance Gap (Cross Entropy - MSE) vs. Network Depth")
plt.ylabel("Accuracy Gap")
plt.xlabel("Network Depth")
plt.axhline(0, color='black', linewidth=0.8)
for i, v in enumerate(gap_values):
    plt.text(i, v + 0.005 if v > 0 else v - 0.01, f"+{v:.4f}", ha='center')
plt.show()
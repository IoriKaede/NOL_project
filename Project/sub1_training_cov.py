import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


from activation_functions import logi, softmax
from data_loader import DataLoader
from models import NeuralNetwork
from supplementary import Value, load_mnist

from four_loss_function import mse_loss, mae_loss, cross_entropy_loss, hinge_loss


TARGET_ACCURACY = 0.80
MAX_EPOCHS_SAFETY = 50
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

train_labels = np.zeros((len(train_y), 10));
train_labels[np.arange(len(train_y)), train_y] = 1
test_labels = np.zeros((len(test_y), 10));
test_labels[np.arange(len(test_y)), test_y] = 1

# DataLoader
train_loader = DataLoader(list(zip(train_images, train_labels)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=BATCH_SIZE, shuffle=True)
train_dataset_size = len(train_images)
test_dataset_size = len(test_images)


loss_functions = {
    "Cross Entropy": cross_entropy_loss,
    "MSE": mse_loss,
    "Hinge": hinge_loss,
    "MAE": mae_loss
}

results = {}

print(f"\n{'=' * 50}")
print(f"starting testing：find {TARGET_ACCURACY * 100}% needed epochs")
print(f"allow max epochs: {MAX_EPOCHS_SAFETY}")
print(f"{'=' * 50}\n")

for name, loss_fn in loss_functions.items():
    print(f"testing: {name} ...")

    nn = NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[logi, logi, logi, softmax]
    )

    epoch = 0
    current_acc = 0.0
    start_time = time.time()
    acc_history = []

    while current_acc < TARGET_ACCURACY and epoch < MAX_EPOCHS_SAFETY:
        epoch += 1

        # Train
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            nn.reset_gradients()
            imgs = np.vstack([x[0] for x in batch])
            lbls = np.vstack([x[1] for x in batch])

            out = nn(Value(imgs, "X"))
            loss = loss_fn(out, Value(lbls, "Y"))
            loss.backward()
            nn.gradient_descent(LEARNING_RATE)

        # Test
        correct = 0
        for batch in test_loader:
            imgs = np.vstack([x[0] for x in batch])
            lbls = np.vstack([x[1] for x in batch])
            out = nn(Value(imgs, "X"))

            pred = np.argmax(out.data, axis=1)
            true = np.argmax(lbls, axis=1)
            correct += np.sum(pred == true)

        current_acc = correct / test_dataset_size
        acc_history.append(current_acc)

        print(f"   -> Epoch {epoch}: Acc = {current_acc:.4f}")


    duration = time.time() - start_time
    if current_acc >= TARGET_ACCURACY:
        print(f" {name} achieve！using time: {epoch} epoch ({duration:.1f}s)\n")
        results[name] = {"epochs": epoch, "time": duration, "status": "Success"}
    else:
        print(f" {name} fail！run {epoch} epoch (final: {current_acc:.4f})\n")
        results[name] = {"epochs": f">{MAX_EPOCHS_SAFETY}", "time": duration, "status": "Failed"}


    plt.plot(range(1, epoch + 1), acc_history, label=name)


plt.axhline(y=TARGET_ACCURACY, color='r', linestyle='--', label='Target 80%')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Convergence Speed Comparison')
plt.legend()
plt.show()


print(f"\n{'=' * 60}")
print(f"{'Loss Function':<20} | {'Epochs needed':<15} | {'Time (s)':<15}")
print(f"{'-' * 60}")
for name, data in results.items():
    print(f"{name:<20} | {str(data['epochs']):<15} | {data['time']:.1f}")
print(f"{'=' * 60}")
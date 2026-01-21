
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from activation_functions import logi, softmax
from data_loader import DataLoader
from four_loss_function import mse_loss, mae_loss, cross_entropy_loss, hinge_loss
from models import NeuralNetwork
from supplementary import Value, load_mnist

np.set_printoptions(precision=2)

data_dir = Path(__file__).resolve().parent / "data"
try:
    train_images, train_y = load_mnist(data_dir, kind='train')
    test_images, test_y = load_mnist(data_dir, kind='t10k')
except Exception as e:
    print("not found")
    raise e

train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

train_labels = np.zeros((len(train_y), 10))
train_labels[np.arange(len(train_y)), train_y] = 1
test_labels = np.zeros((len(test_y), 10))
test_labels[np.arange(len(test_y)), test_y] = 1

# DataLoader
train_dataset = list(zip(train_images, train_labels))
test_dataset = list(zip(test_images, test_labels))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False)
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

learning_rate = 0.1
epochs = 10


loss_functions_dict = {
    "MSE": mse_loss,
    "MAE": mae_loss,
    "Cross Entropy": cross_entropy_loss,
    "Hinge": hinge_loss
}


for loss_name, loss_func in loss_functions_dict.items():
    print(f"\n{'=' * 20}\ntraining: {loss_name}\n{'=' * 20}")

    neural_network = NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[logi, logi, logi, softmax]
    )

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # 2.
    for epoch in range(1, epochs + 1):
        # --- Training ---
        train_loss = 0.0
        correctly_classified = 0

        for batch in tqdm(train_loader, desc=f"[{loss_name}] Epoch {epoch} Train"):
            neural_network.reset_gradients()

            images_np = np.vstack([img for img, _ in batch])
            labels_np = np.vstack([lbl for _, lbl in batch])
            images = Value(images_np, expr="X")
            labels = Value(labels_np, expr="Y")

            output = neural_network(images)

            loss = loss_func(output, labels)

            loss.backward()
            neural_network.gradient_descent(learning_rate)

            train_loss += loss.data
            true_cls = np.argmax(labels_np, axis=1)
            pred_cls = np.argmax(output.data, axis=1)
            correctly_classified += np.sum(true_cls == pred_cls)

        train_losses.append(train_loss / train_dataset_size)
        train_accuracies.append(correctly_classified / train_dataset_size)

        # --- Testing ---
        test_loss = 0.0
        correctly_classified_test = 0
        for batch in tqdm(test_loader, desc=f"[{loss_name}] Epoch {epoch} Test"):
            images_np = np.vstack([img for img, _ in batch])
            labels_np = np.vstack([lbl for _, lbl in batch])
            images = Value(images_np, expr="X")
            labels = Value(labels_np, expr="Y")

            output = neural_network(images)
            loss = loss_func(output, labels)
            test_loss += loss.data

            true_cls = np.argmax(labels_np, axis=1)
            pred_cls = np.argmax(output.data, axis=1)
            correctly_classified_test += np.sum(true_cls == pred_cls)

        test_losses.append(test_loss / test_dataset_size)
        test_accuracies.append(correctly_classified_test / test_dataset_size)

        print(f"Epoch {epoch}: Train Acc: {train_accuracies[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

    # 3.

    plt.figure()
    plt.title(f"Loss ({loss_name}): train vs test")
    plt.semilogy(range(1, epochs + 1), train_losses, label="train")
    plt.semilogy(range(1, epochs + 1), test_losses, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f"Accuracy ({loss_name}): train vs test")
    plt.plot(range(1, epochs + 1), train_accuracies, label="train")
    plt.plot(range(1, epochs + 1), test_accuracies, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    print(f"Finished {loss_name}.\n")

print("done")
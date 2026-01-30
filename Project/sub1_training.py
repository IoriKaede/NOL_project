import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from activation_functions import logi, softmax, relu
from data_loader import DataLoader
from four_loss_function import mse_loss, cross_entropy_loss, mae_loss, hinge_loss
from models import NeuralNetwork
from supplementary import Value, load_mnist

np.set_printoptions(precision=2)
data_dir = Path(__file__).resolve().parent / "data"
train_images, train_y = load_mnist(data_dir, kind='train')
test_images, test_y = load_mnist(data_dir, kind='t10k')

train_images = train_images.reshape(60_000, 784) / 255
test_images = test_images.reshape(10_000, 784) / 255

train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1
test_labels = np.zeros((10_000, 10))
test_labels[np.arange(10_000), test_y] = 1

validation_subset = 5000
validation_images = train_images[:validation_subset]
validation_labels = train_labels[:validation_subset]
train_images = train_images[validation_subset:]
train_labels = train_labels[validation_subset:]

train_dataset = list(zip(train_images, train_labels))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
train_dataset_size = len(train_dataset)

validation_dataset = list(zip(validation_images, validation_labels))
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, drop_last=False)
validation_dataset_size = len(validation_dataset)

neural_network = NeuralNetwork(
    layers=[784, 256, 128, 64, 10],
    activation_functions=[relu, relu, relu, softmax]
)

learning_rate = 0.01
epochs = 200

train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss = 0.0
    correctly_classified = 0
    num_batch = 0
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        num_batch += 1
        neural_network.reset_gradients()
        images = np.vstack([image for (image, _) in batch])
        labels = np.vstack([label for (_, label) in batch])
        images = Value(images, expr="X")
        labels = Value(labels, expr="Y")

        output = neural_network(images)
        loss = mse_loss(output, labels)
        loss.backward()
        neural_network.gradient_descent(learning_rate)

        train_loss += loss.data
        true_classification = np.argmax(labels.data, axis=1)
        predicted_classification = np.argmax(output.data, axis=1)
        correctly_classified += np.sum(true_classification == predicted_classification)

    train_losses.append(train_loss / num_batch)
    train_accuracies.append(correctly_classified / train_dataset_size)

    val_loss_epoch = 0.0
    val_correct = 0
    num_val_batch = 0
    for batch in validation_loader:
        num_val_batch += 1
        images_v = np.vstack([image for (image, _) in batch])
        labels_v = np.vstack([label for (_, label) in batch])
        out_v = neural_network(Value(images_v))
        val_loss_epoch += mse_loss(out_v, Value(labels_v)).data
        val_correct += np.sum(np.argmax(labels_v, axis=1) == np.argmax(out_v.data, axis=1))

    validation_losses.append(val_loss_epoch / num_val_batch)
    validation_accuracies.append(val_correct / validation_dataset_size)
    print(f"Epoch {epoch}: Train Acc {train_accuracies[-1]:.4f}, Val Acc {validation_accuracies[-1]:.4f}")


def find_overfitting_point(epochs_count, data_list, degree=3):

    x = np.array(range(1, epochs_count + 1))
    y = np.array(data_list)

    coef = np.polyfit(x, y, degree)
    poly = np.poly1d(coef)

    derivative = poly.deriv()

    roots = derivative.roots

    real_roots = roots[np.isreal(roots)].real
    valid_roots = [r for r in real_roots if 1 <= r <= epochs_count]

    min_point = None
    for r in valid_roots:
        if poly.deriv(2)(r) > 0:
            min_point = r
            break

    return min_point, poly


def plot_with_fitting(epochs_count, train_data, val_data, title, ylabel, degree=3, detect_overfit=False):
    x = np.array(range(1, epochs_count + 1))

    plt.figure(figsize=(10, 6))
    plt.title(title)

    plt.plot(x, train_data, 'b--', alpha=0.3, label="Train (Original)")
    plt.plot(x, val_data, 'r--', alpha=0.3, label="Val (Original)")

    train_poly = np.poly1d(np.polyfit(x, train_data, degree))
    plt.plot(x, train_poly(x), 'b-', linewidth=2, label="Train (Fitted)")

    overfit_epoch, val_poly = find_overfitting_point(epochs_count, val_data, degree)
    plt.plot(x, val_poly(x), 'r-', linewidth=2, label="Val (Fitted)")

    if detect_overfit and overfit_epoch:
        val_loss_at_min = val_poly(overfit_epoch)
        plt.scatter(overfit_epoch, val_loss_at_min, color='green', s=100, zorder=5,
                    label=f'Overfitting Point (Epoch {overfit_epoch:.2f})')
        plt.axvline(x=overfit_epoch, color='green', linestyle=':', alpha=0.7)
        print(f"[{title}] overfitting {overfit_epoch:.2f} epoch。")

    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

plot_with_fitting(epochs, train_losses, validation_losses,
                  "MSE Loss: Train vs Validation", "Loss", degree=3, detect_overfit=True)

plot_with_fitting(epochs, train_accuracies, validation_accuracies,
                  "MSE Accuracy: Train vs Validation", "Accuracy", degree=3, detect_overfit=False)

plt.show()
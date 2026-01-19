from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # gives progression bars when running code

from activation_functions import logi, softmax
from data_loader import DataLoader
from loss_functions import mse_loss,mae_loss,cross_entropy_loss, hinge_loss
from models import NeuralNetwork
from supplementary import Value, load_mnist


# Set printing precision for NumPy so that we don't get needlessly many digits in our answers.
np.set_printoptions(precision=2)

# Get images and corresponding labels from the (fashion-)mnist dataset
data_dir = Path(__file__).resolve().parent / "data"
train_images, train_y = load_mnist(data_dir, kind='train')
test_images, test_y = load_mnist(data_dir, kind='t10k')

# Reshape each of the 60 000 images from a 28x28 image into a 784 vector.
# Rescale the values in the 784 to be in [0,1] instead of [0, 255].
train_images = train_images.reshape(60_000, 784) / 255
test_images = test_images.reshape(10_000, 784) / 255

# Labels are stored as numbers. For neural network training, we want one-hot encoding, i.e. the label should be a vector
# of 10 long with a one in the index corresponding to the digit.
train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1
test_labels = np.zeros((10_000, 10))
test_labels[np.arange(10_000), test_y] = 1

# We create our own validation set by placing the first 5000 images in the validation dataset and kepping the rest in
# the training set.
validation_subset = 5000
validation_images = train_images[:validation_subset]
validation_labels = train_labels[:validation_subset]
train_images = train_images[validation_subset:]
train_labels = train_labels[validation_subset:]

# The data loader takes at every iteration batch_size items from the dataset. If it is not possible to take batch_size
# items, it takes whatever it still can. With a dataset of 100 images and a batch size of 32, it will be batches of
# 32, 32, 32, and 4.
train_dataset = list(zip(train_images, train_labels))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
train_dataset_size = len(train_dataset)

validation_dataset = list(zip(validation_images, validation_labels))
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, drop_last=False)
validation_dataset_size = len(validation_dataset)

test_dataset = list(zip(test_images, test_labels))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False)
test_dataset_size = len(test_dataset)

# Initialize a neural network with some layers and the default activation functions.
neural_network = NeuralNetwork(
    layers=[784, 256, 128, 64, 10],
    activation_functions=[logi, logi, logi, softmax]
)
# OR load the parameters of some other trained network from disk
# neural_network = NeuralNetwork(
#   layers=[784, 256, 128, 64, 10],
#   activation_functions=[logi, logi, logi, softmax]
# ).load("path/to/some/folder")

# Set training configuration
learning_rate = 3e-3
epochs = 1 #change epoch here <-

results = {}
loss_functions = { #loss fct setting
    "MSE": mse_loss,
    "MAE": mae_loss,
    "Cross-Entropy": cross_entropy_loss,
    "Hinge": hinge_loss
}


for name, loss_fn in loss_functions.items():

    net = NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[logi, logi, logi, softmax]
    )
    history = {'train_loss': [], 'test_acc': []}
    lr = 0.001  #standardize

    for epoch in range(epochs):  # epoch numbers
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            net.reset_gradients()

            images = Value(np.vstack([b[0] for b in batch]))
            labels = Value(np.vstack([b[1] for b in batch]))

            output = net(images)
            loss = loss_fn(output, labels)

            loss.backward()
            net.gradient_descent(lr)

            # should we calculate average? may fix the problem of loss gap
            epoch_loss += loss.data

        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        correct = 0
        for batch in test_loader:
            imgs = Value(np.vstack([b[0] for b in batch]))
            b = np.vstack([b[1] for b in batch])
            out = net(imgs)
            correct += np.sum(np.argmax(out.data, axis=1) == np.argmax(b, axis=1))

        acc = correct / test_dataset_size #accuary check
        history['test_acc'].append(acc)
        print(f"{name} - accuarcy: {acc:.4f}, average loss: {avg_loss:.4f}")

    results[name] = history



plt.figure()

for name in results:
    plt.plot(results[name]['train_loss'], label=name)
plt.title("Convergence comparison")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()

plt.figure()
for name in results:
    plt.plot(results[name]['test_acc'], label=name)
plt.title("Accuracy comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()



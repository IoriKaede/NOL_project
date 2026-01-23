from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # gives progression bars when running code

from activation_functions import logi, softmax, relu
from data_loader import DataLoader
from loss_functions import mse_loss,mae_loss, cross_entropy_loss,hinge_loss
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

loss_functions = { #loss function dic
    "MSE": mse_loss,
    "MAE": mae_loss,
    "CrossEntropy": cross_entropy_loss,
    "Hinge": hinge_loss
}
#for comparison
all_train_losses = {}
all_val_losses = {}
all_train_accuracies = {}
all_val_accuracies = {}
# Set training configuration
learning_rate = 0.01
epochs = 50
# Initialize a neural network with some layers and the default activation functions.

for loss_name, loss_fct in loss_functions.items():
    print(f"{loss_name}")
    neural_network = NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[relu, relu, relu, softmax]
    )

    # OR load the parameters of some other trained network from disk
    # neural_network = NeuralNetwork(
    #   layers=[784, 256, 128, 64, 10],
    #   activation_functions=[logi, logi, logi, softmax]
    # ).load("path/to/some/folder")

    # Do the full training algorithm
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    for epoch in range(1, epochs + 1):
        # (Re)set the training loss for this epoch.
        train_loss = 0.0
        correctly_classified = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            # Reset the gradients so that we start fresh.
            neural_network.reset_gradients()

            # Get the images and labels from the batch
            images = np.vstack([image for (image, _) in batch])
            labels = np.vstack([label for (_, label) in batch])

            # Wrap images and labels in a Value class.
            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            # Compute what the model says is the label.
            output = neural_network(images)

            # Compute the loss for this batch.
            loss = loss_fct(
                output,
                labels
            )

            # Do backpropagation
            loss.backward()

            # Update the weights and biases using the chosen algorithm, in this case gradient descent.
            neural_network.gradient_descent(learning_rate)

            # Store the loss for this batch.
            train_loss += loss.data

            # Store accuracies for extra interpretability
            true_classification = np.argmax(
                labels.data,
                axis=1
            )
            predicted_classification = np.argmax(
                output.data,
                axis=1
            )
            correctly_classified += np.sum(true_classification == predicted_classification)

        # Store the loss and average accuracy for the entire epoch.
        train_losses.append(train_loss / train_dataset_size)
        train_accuracies.append(correctly_classified / train_dataset_size)

        print(f"Accuracy: {train_accuracies[-1]}")
        print(f"Loss: {train_loss}")
        print("")

        validation_loss = 0.0
        correctly_classified = 0
        for batch in tqdm(validation_loader, desc=f"Validation epoch {epoch}"):
            # Get the images and labels from the batch
            images = np.vstack([image for (image, _) in batch])
            labels = np.vstack([label for (_, label) in batch])

            # Wrap images and labels in a Value class.
            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            # Compute what the model says is the label.
            output = neural_network(images)

            # Compute the loss for this batch.
            loss = loss_fct(
                output,
                labels
            )

            # Store the loss for this batch.
            validation_loss += loss.data

            # Store accuracies for extra interpretability
            true_classification = np.argmax(
                labels.data,
                axis=1
            )
            predicted_classification = np.argmax(
                output.data,
                axis=1
            )
            correctly_classified += np.sum(true_classification == predicted_classification)

        validation_losses.append(validation_loss / validation_dataset_size)
        validation_accuracies.append(correctly_classified / validation_dataset_size)

        print(f"Accuracy: {validation_accuracies[-1]}")
        print(f"Loss: {validation_loss}")
        print("")

        # add storage
        all_train_losses[loss_name] = train_losses.copy()
        all_val_losses[loss_name] = validation_losses.copy()
        all_train_accuracies[loss_name] = train_accuracies.copy()
        all_val_accuracies[loss_name] = validation_accuracies.copy()

    print(" === SUMMARY === ")
    print(" --- training --- ")
    print(f"Accuracies: {train_accuracies}")
    print(f"Losses: {train_losses}")
    print("")
    print(" --- validation --- ")
    print(f"Accuracies: {validation_accuracies}")
    print(f"Losses: {validation_losses}")
    print("")

    plt.figure()
    plt.title("Validation Loss Convergence Comparison")
    for loss_name, losses in all_val_losses.items():
        plt.semilogy(range(1, epochs + 1), losses, label=loss_name)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()

    # Accuracy convergence
    plt.figure()
    plt.title("Validation Accuracy Comparison")
    for loss_name, accs in all_val_accuracies.items():
        plt.plot(range(1, epochs + 1), accs, label=loss_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Final accuracy bar plot
    plt.figure()
    plt.title("Final Validation Accuracy Comparison")
    names = list(all_val_accuracies.keys())
    final_accs = [all_val_accuracies[n][-1] for n in names]
    plt.bar(names, final_accs)
    plt.ylabel("Accuracy")

    plt.show()
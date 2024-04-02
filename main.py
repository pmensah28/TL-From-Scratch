from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
from mnist_nn import NeuralNetwork

# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to fit the model (flattening the images)
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Filter the dataset for odd numbers (training)
odd_indices_train = np.where(Y_train % 2 != 0)[0]
odd_indices_test = np.where(Y_test % 2 != 0)[0]

X_train_odd = X_train[odd_indices_train]
Y_train_odd = Y_train[odd_indices_train]
X_test_odd = X_test[odd_indices_test]
Y_test_odd = Y_test[odd_indices_test]

# Filter the dataset for even numbers (transfer learning)
even_indices_train = np.where(Y_train % 2 == 0)[0]
even_indices_test = np.where(Y_test % 2 == 0)[0]

X_train_even = X_train[even_indices_train]
Y_train_even = Y_train[even_indices_train]
X_test_even = X_test[even_indices_test]
Y_test_even = Y_test[even_indices_test]

# Convert labels to one-hot encoded format
no_classes = 10
Y_train_odd_one_hot = to_categorical(Y_train_odd, no_classes)
Y_test_odd_one_hot = to_categorical(Y_test_odd, no_classes)
Y_train_even_one_hot = to_categorical(Y_train_even, no_classes)
Y_test_even_one_hot = to_categorical(Y_test_even, no_classes)

# Implementation for odd Mnist
nn_odd = NeuralNetwork(input_layer=784, hidden_layer=500, output_layer=10, learning_rate=0.0001)

n_epochs = 200
print("Training for Mnist_odd starts")
nn_odd.fit(X_train_odd, Y_train_odd_one_hot, X_test_odd, Y_test_odd_one_hot, n_epochs)

print("Finished Training for Mnist_odd")
y_mnist_odd_pred = nn_odd.predict(X_test_odd)

accuracy = nn_odd.accuracy(Y_test_odd_one_hot, y_mnist_odd_pred)
print(" ")
print("Accuracy (Mnist-Odd):", accuracy)
print(" ")
# Transfer Learning
n_epochs = 100
print("Transfer learning for Mnist_Even using Mnist_odd features starts")
nn_odd.fit(X_train_even, Y_train_even_one_hot, X_test_even, Y_test_even_one_hot, n_epochs, freeze_layer = True)
print("Finished Training transfer learning Mnist_even using Mnist_odd features")

y_mnist_odd_TL_Even_pred = nn_odd.predict(X_test_even)
TL_even_accuracy = nn_odd.accuracy(Y_test_even_one_hot, y_mnist_odd_TL_Even_pred)
print(" ")
print("Accuracy (TL_Mnist-Odd):", TL_even_accuracy)
print(" ")

# Implementation for even Mnist
nn_even = NeuralNetwork(input_layer=784, hidden_layer=500, output_layer=10, learning_rate=0.0001)

n_epochs = 200
print("Training for Mnist_even starts")
nn_even.fit(X_train_even, Y_train_even_one_hot, X_test_even, Y_test_even_one_hot, n_epochs)

y_mnist_Even_pred = nn_even.predict(X_test_even)
even_accuracy = nn_even.accuracy(Y_test_even_one_hot,  y_mnist_Even_pred)
print(" ")
print("Accuracy (Mnist-even):", even_accuracy)
print(" ")

# Transfer learning
print("Transfer learning for Mnist_odd using Mnist_even features start")
n_epochs = 100
nn_even.fit(X_train_odd, Y_train_odd_one_hot, X_test_odd, Y_test_odd_one_hot, n_epochs, freeze_layer = True)
print("Finished Training transfer learning Mnist_odd using Mnist_even features")

y_mnist_Even_TL_odd_pred = nn_even.predict(X_test_odd)
Even_TL_odd_accuracy = nn_even.accuracy(Y_test_odd_one_hot, y_mnist_Even_TL_odd_pred)
print(" ")
print("Accuracy (TL_Mnist_Even_accuracy):", Even_TL_odd_accuracy)
print(" ")

plt.figure(figsize=(5, 3))
plt.plot(nn_odd.train_loss, label='Training Loss')
plt.plot(nn_odd.test_loss, label='Validation Loss')
plt.title('Loss over Epochs - Mnist_odd')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(nn_odd.train_loss, label='Training Loss')
plt.plot(nn_odd.test_loss, label='Validation Loss')
plt.title('Loss over Epochs TL for Mnist_Even using Mnist_odd features start')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Finished Training for Mnist_even")
plt.figure(figsize=(5, 3))
plt.plot(nn_even.train_loss, label='Training Loss')
plt.plot(nn_even.test_loss, label='Validation Loss')
plt.title('Loss over Epochs - Mnist_even')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(nn_even.train_loss, label='Training Loss')
plt.plot(nn_even.test_loss, label='Validation Loss')
plt.title('Loss over Epochs - TL for Mnist_odd using Mnist_even features start')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
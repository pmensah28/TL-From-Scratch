import numpy as np


class MnistNNClassifier:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.W1, self.W2, self.b1, self.b2 = self.init_params()

    def init_params(self):
        W1 = np.random.randn(self.hidden_layer, self.input_layer) * np.sqrt(2 / (self.input_layer + self.hidden_layer))
        W2 = np.random.randn(self.output_layer, self.hidden_layer) * np.sqrt(
            2 / (self.hidden_layer + self.output_layer))
        b1 = np.zeros((self.hidden_layer, 1))
        b2 = np.zeros((self.output_layer, 1))
        return W1, W2, b1, b2

    def relu(self, z):
        return np.maximum(0, z)

    def d_relu(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        expZ = np.exp(z - np.max(z))
        return expZ / expZ.sum(axis=0, keepdims=True)
    #
    # def cross_entropy_loss(self, Y, A2):
    #     m = Y.shape[1]
    #     epsilon = 1e-7  # Small value to prevent log(0)
    #     log_probs = -np.log(A2 + epsilon)
    #     loss = np.sum(Y * log_probs) / m
    #     return loss

    def cross_entropy_loss(self, Y, A2):
        m = Y.shape[0]  # Number of examples
        epsilon = 1e-7  # Small value to prevent log(0)
        A2 = A2.T  # Transpose A2 to match Y's shape
        log_probs = -np.log(A2 + epsilon)
        loss = np.sum(Y * log_probs) / m
        return loss

    def forward_pass(self, X):
        Z1 = np.dot(self.W1, X.T) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)

        return A2, Z2, A1, Z1

    def backward_pass(self, X, Y, A2, Z2, A1, Z1):
        m = Y.shape[1]

        dZ2 = A2 - Y.T
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * self.d_relu(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, dW2, db1, db2

    def update_params(self, dW1, dW2, db1, db2):
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2

    def predict(self, X):
        A2, _, _, _ = self.forward_pass(X)
        return np.argmax(A2, axis=0)

    def accuracy(self, Y, Y_pred):
        return np.mean(Y.argmax(axis=0) == Y_pred)

    def fit(self, X_train, Y_train, X_test, Y_test, n_epochs):
        self.train_loss, self.test_loss = [], []
        for i in range(n_epochs):
            A2, Z2, A1, Z1 = self.forward_pass(X_train)
            dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1)
            self.update_params(dW1, dW2, db1, db2)
            self.train_loss.append(self.cross_entropy_loss(Y_train, A2))
            A2_test, _, _, _ = self.forward_pass(X_test)
            self.test_loss.append(self.cross_entropy_loss(Y_test, A2_test))
            if i % 10 == 0:  # Print loss every 10 epochs
                print(f"Epoch {i}, Train Loss: {self.train_loss[-1]}, Test Loss: {self.test_loss[-1]}")

import numpy as np

class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.W1, self.W2, self.b1, self.b2 = self.init_params()

    def init_params(self):
        W1 = np.random.randn(self.hidden_layer, self.input_layer) * np.sqrt(2 / (self.input_layer + self.hidden_layer))
        W2 = np.random.randn(self.output_layer, self.hidden_layer) * np.sqrt(2 / (self.hidden_layer + self.output_layer))
        b1 = np.zeros((self.hidden_layer, 1))
        b2 = np.zeros((self.output_layer, 1))
        return W1, W2, b1, b2

    def relu(self, z):
        return np.maximum(0, z)

    def d_relu(self, z):
        return (z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def cross_entropy_loss(self, Y, A2):
        m = Y.shape[0]
        A2 = A2.T
        loss = -np.sum(Y * np.log(A2)) / m
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
        dW1 = (1 / m) * np.dot(dZ1, X)  # Corrected from np.dot(dZ1, X)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, dW2, db1, db2

    def update_params(self, dW1, dW2, db1, db2, freeze_layer = False):
        if freeze_layer == False:
          self.W1 -= self.learning_rate * dW1
          self.b1 -= self.learning_rate * db1

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def predict(self, X):
        A2, _, _, _ = self.forward_pass(X)
        predictions = np.argmax(A2, axis=0)  # Make sure axis aligns with how A2 is structured
        return predictions

    def accuracy(self, Y, Y_pred):
        # Convert one-hot encoded Y to class labels if it's one-hot encoded
        if Y.shape[1] > 1:  # More than one column implies one-hot encoding
            Y = np.argmax(Y, axis=1)
        # Calculate accuracy
        accuracy = np.mean(Y_pred == Y) * 100
        return accuracy

    def fit(self, X_train, Y_train, X_test, Y_test, n_epochs, freeze_layer = False):
        self.train_loss, self.test_loss = [], []
        for i in range(n_epochs):
            A2, Z2, A1, Z1 = self.forward_pass(X_train)
            dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1)
            self.update_params(dW1, dW2, db1, db2, freeze_layer)
            self.train_loss.append(self.cross_entropy_loss(Y_train, A2))
            A2_test, _, _, _ = self.forward_pass(X_test)
            self.test_loss.append(self.cross_entropy_loss(Y_test, A2_test))
            if i % 1 == 0:  # Print loss every 10 epochs
                print(f"Epoch {i+1}, Train Loss: {self.train_loss[-1]}, Test Loss: {self.test_loss[-1]}")

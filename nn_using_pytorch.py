import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

### Gets the GPU if there is one, otherwise the cpu"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Softmax_reg(nn.Module):
  def __init__(self):
    super(Softmax_reg, self).__init__()
    self.fc1 = nn.Linear(28*28, 10)
    self.epochs = 100
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    self.total = 0
    self.correct = 0

  def forward(self, x):
    x = x.view(-1, 28*28)
    # x = nn.functional.softmax(x)
    x = self.fc1(x)
    return x

  def fit(self, trainloader, testloader):
    self.train_losses = []
    self.test_losses = []

    #Trainning step
    for epoch in range(self.epochs):
      self.train()
      train_running_loss = 0.0
      for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        self.optimizer.zero_grad()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        train_running_loss += loss.item()
      self.train_losses.append(train_running_loss / len(trainloader))

      # Validation step
      self.eval()
      test_running_loss = 0.0
      for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        test_running_loss += loss.item()
      self.test_losses.append(test_running_loss / len(testloader))

      print(f"Epoch {epoch+1}, Train Loss: {self.train_losses[-1]}, Test Loss: {self.test_losses[-1]}")
  
  #Function to plot losses
  def plot_losses(self,name):
    plt.plot(self.train_losses, label='Train Loss')
    plt.plot(self.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig(f'loss_softmax_reg_{name}_plot.png')
    plt.show()

  # Function for testing step
  def test(self,testloader):
    self.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
      for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = self(images)
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
    print(f"Accuracy on odd numbers: {100 * self.correct / self.total}%")

  # Function to plot confusion matrix
  def plot_confusion_matrix(self, testloader,name):
    self.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation for validation
      for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = self(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_even = [0,2,4,6,8]
    class_odd = [1,3,5,7,9]
    if name == "odd":
      labels = class_odd
    elif name == "even_odd":
      labels = class_odd
    elif name == "even":
      labels = class_even
    else:
      labels = class_even

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'CM_softmax_reg_{name}.png')
    plt.show()

class Mnist(nn.Module):
  def __init__(self):
    super(Mnist, self).__init__()
    self.fc1 = nn.Linear(28*28, 500)
    self.fc2 = nn.Linear(500, 10)
    self.epochs = 100
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    self.total = 0
    self.correct = 0

  def forward(self, x):
    x = x.view(-1, 28*28)
    x = nn.functional.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def fit(self, trainloader, testloader):
    self.train_losses = []
    self.test_losses = []

    #Training step
    for epoch in range(self.epochs):
      self.train()
      train_running_loss = 0.0
      for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        self.optimizer.zero_grad()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        train_running_loss += loss.item()
      self.train_losses.append(train_running_loss / len(trainloader))

      # Validation step
      self.eval()
      test_running_loss = 0.0
      for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        test_running_loss += loss.item()
      self.test_losses.append(test_running_loss / len(testloader))
      print(f"Epoch {epoch+1}, Train Loss: {self.train_losses[-1]}, Test Loss: {self.test_losses[-1]}")
  
  #Function  to plot losses
  def plot_losses(self,name):
    plt.plot(self.train_losses, label='Train Loss')
    plt.plot(self.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig(f'loss_{name}_plot.png')
    plt.show()

  #Function for testing step
  def test(self,testloader):
    self.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
      for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = self(images)
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
    print(f"Accuracy on odd numbers: {100 * self.correct / self.total}%")

# Function to plot confusion matrix
  def plot_confusion_matrix(self, testloader,name):
    self.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation for validation
      for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = self(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_even = [0,2,4,6,8]
    class_odd = [1,3,5,7,9]
    if name == "odd":
      labels = class_odd
    elif name == "even_odd":
      labels = class_odd
    elif name == "even":
      labels = class_even
    else:
      labels = class_even

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'CM_{name}.png')
    plt.show()


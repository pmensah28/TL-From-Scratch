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
print(device)

class Mnist(nn.Module):
  def __init__(self):
    super(Mnist, self).__init__()
    self.fc1 = nn.Linear(28*28, 500)
    self.fc2 = nn.Linear(500, 10)
    self.epochs = 100
    self.correct = 0
    self.total = 0
    self.criterion = nn.CrossEntropyLoss() #nn.NLLLoss(reduction='sum')
    self.optimizer = optim.SGD(self.parameters(), lr=0.001,momentum=.90)#optim.Adam(self.parameters(), lr = .01)


  def forward(self, x):
    x = x.view(-1, 28*28)  # Flatten the image
    x = F.relu(self.fc1(x))
    x = self.fc2(x)#F.relu(self.fc2(x))
    return x#F.log_softmax(x, dim=1)

  def fit(self,trainloader):
    self.losses = []
    for epoch in range(self.epochs):
      self.train() # Set the model to training mode
      running_loss = 0.0
      for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        self.optimizer.zero_grad()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
      print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")
      self.losses.append(running_loss / len(trainloader))

  def test(self,testloader):
    # Validation step
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
  def plot_confusion_matrix(self, testloader):
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

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

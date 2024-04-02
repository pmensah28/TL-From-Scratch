# Importation
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
import torch.optim as optim
from nn_using_pytorch import Softmax_reg
from nn_using_pytorch import Mnist

# Gets the GPU if there is one, otherwise the cpu"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""Data Splitting"""
mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

# Separating odd and even digits
train_odd = [i for i in range(len(mnist_trainset)) if mnist_trainset.targets[i] % 2 != 0]
train_even = [i for i in range(len(mnist_trainset)) if mnist_trainset.targets[i] % 2 == 0]
test_odd = [i for i in range(len(mnist_testset)) if mnist_testset.targets[i] % 2 != 0]
test_even = [i for i in range(len(mnist_testset)) if mnist_testset.targets[i] % 2 == 0]

trainset_odd = Subset(mnist_trainset, train_odd)
trainloader_odd = DataLoader(trainset_odd, batch_size=64, shuffle=True)
testset_odd = Subset(mnist_testset, test_odd)
testloader_odd = DataLoader(testset_odd, batch_size=64, shuffle=True)

trainset_even = Subset(mnist_trainset, train_even)
trainloader_even = DataLoader(trainset_even, batch_size=64, shuffle=True)
testset_even = Subset(mnist_testset, test_even)
testloader_even = DataLoader(testset_even, batch_size=64, shuffle=True)


'''Task1: Tain both dataset using SoftMax Regression '''

 # Creating, training and testing SoftMax Regression model to train even dataset
model_softmax_reg_even = Softmax_reg()
model_softmax_reg_even.to(device)

#Training on even dataset
model_softmax_reg_even.fit(trainloader_even,testloader_even) 

#Ploting the losses
model_softmax_reg_even.plot_losses("even")

# Testing step
model_softmax_reg_even.test(testloader_even)

#Ploting confusion matrix
model_softmax_reg_even.plot_confusion_matrix(testloader_even,"even")

#Creating, training and testing SoftMax Regression model to train odd dataset
model_softmax_reg_odd = Softmax_reg()
model_softmax_reg_odd.to(device)

#Training on odd dataset
model_softmax_reg_odd.fit(trainloader_odd,testloader_odd) 

#Ploting the losses
model_softmax_reg_odd.plot_losses("odd")

# Testing step
model_softmax_reg_odd.test(testloader_odd)

#Ploting confusion matrix
model_softmax_reg_odd.plot_confusion_matrix(testloader_odd,"odd")


'''Task2: Tain both dataset using 1-NN (! hidden layer Neural Network)'''

#Creating, training and testing 1-nn model to train even dataset 
model_even = Mnist()
model_even.to(device)

#Training on even dataset
model_even.fit(trainloader_even, testloader_even) #Training on even dataset

#Ploting the losses
model_even.plot_losses("even")

# Testing step
model_even.test(testloader_even)

#Ploting confusion matrix
model_even.plot_confusion_matrix(testloader_even,"even")


#Creating, training and testing 1-nn model to train odd dataset 
model_odd = Mnist().to(device)

#Training on odd dataset
model_odd.fit(trainloader_odd,testloader_odd) 

#Ploting losses
model_odd.plot_losses('odd')

# Testing step
model_odd.test(testloader_odd)

#Ploting confusion matrix
model_odd.plot_confusion_matrix(testloader_odd,'odd')


'''Task3: Tain both dataset using Transfert Learning'''

#Freezing weigth and bias of hidden layer in odd model and used it to train even dataset
for param in model_odd.fc1.parameters():
  param.requires_grad = False
model_odd.optimizer = optim.Adam(model_odd.parameters(), lr=.001)
model_odd.epochs = 30

#Training on even dataset using odd model
model_odd.fit(trainloader_even,testloader_even)

#Plotting losses step
model_odd.plot_losses('odd_even')

#Testing step
model_odd.test(testloader_even)

#Ploting confusion matrix
model_odd.plot_confusion_matrix(testloader_even,'odd_even')


#Freezing weigth and bias of hidden layer in even model and used it to train odd dataset
for param in model_even.fc1.parameters():
  param.requires_grad = False
model_even.optimizer = optim.Adam(model_even.parameters(), lr=.001)
model_even.epochs = 30

#Training on odd dataset using even model
model_even.fit(trainloader_odd, testloader_odd)

#Ploting losses step
model_even.plot_losses("even_odd")

#Testing step
model_even.test(testloader_odd)

#Plotting confusion matrix
model_even.plot_confusion_matrix(testloader_odd,"even_odd")


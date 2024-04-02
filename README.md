# Transfer Learning from Scratch

This repository contains the implementation of a basic neural network model to demonstrate transfer learning using the mnist dataset with 3 different approach: from scratch, using pytorch, a softmax regression. 
The project is structured to first train a model on odd numbers (1, 3, 5, 7, 9) and then apply transfer learning techniques to adapt the model to recognize even numbers (0, 2, 4, 6, 8) and vice versa.

## Project Overview

Transfer learning is a machine learning technique where a model trained on one task with knowledge saved is transferred to a second related task.
It is an important approach in deep learning because it allows for leveraging pre-trained models, saving on resources and speeds up the training time. 
This project aims to provide a clear example of implementing transfer learning from scratch, without relying on pre-built deep learning frameworks' high-level abstractions.
And we then compare the results with a softmax regression and a pre-built deep learning model like Pytorch.

## Installation

To run this project, you will need Python 3 and the following Python libraries installed:

- NumPy
- TensorFlow or Keras (for loading the MNIST dataset)

You can install the requirements with the following command:

```bash
pip install numpy tensorflow
```


To train the model and apply transfer learning, follow theses steps:

    $ https://github.com/pmensah28/TL-From-Scratch.git
    $ cd Basic-ML-Algorithms
    $ python3 main.py
    $ python3 pytorch_main.py


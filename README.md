# 🧠 IrisNet : Neural Network in C++ for Iris Classification

![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Neural Network](https://img.shields.io/badge/Neural_Network-FF6F61?style=for-the-badge&logo=tensorflow&logoColor=white)
![Iris Dataset](https://img.shields.io/badge/Iris_Dataset-4B8BBE?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Implementation Details](#implementation-details)
7. [Code Explanation](#code-explanation)
8. [Resources](#resources)
9. [Contributing](#contributing)
10. [License](#license)

## 🌟 Introduction

This project implements a simple neural network from scratch in C++ to classify iris flowers using the famous Iris dataset. The neural network is trained to distinguish between three species of iris flowers based on four features: sepal length, sepal width, petal length, and petal width.
![Training](https://github.com/HimanshuMohanty-Git24/IrisNet/assets/94133298/982748b1-e912-46a1-bdad-d7f688fac5fd)   ![CLI app](https://github.com/HimanshuMohanty-Git24/IrisNet/assets/94133298/b7408da1-7aa6-4861-8cb6-02dc3f5a0dc5)

## 🚀 Features

- 🧮 Implementation of a basic feedforward neural network
- 📊 Training on the Iris dataset
- 🎨 Colorful and interactive CLI interface
- 📈 Evaluation metrics including accuracy, confusion matrix, precision, recall, and F1-score
- 💾 Model saving and loading functionality

## 🛠️ Requirements

- C++ compiler with C++11 support
- Standard C++ libraries

## 📥 Installation

1. Clone the repository:
   ```
   git clone https://github.com/HimanshuMohanty-Git24/Neural-Network-CPP.git
   ```
2. Navigate to the project directory:
   ```
   cd Neural-Network-CPP
   ```

## 🖥️ Usage

1. Compile the code:
   ```
   g++ -std=c++11 train.cpp -o train
   ```
2. Run the program:
   ```
   ./train
   ```
3. The program will automatically load the Iris dataset, train the neural network, and display the results.
4. Compile the cli app:
   ```
   g++ -std=c++11 predict.cpp -o predict
   ```
5. Run the cli app
      ```
   ./predict
   ```
## 🧠 Implementation Details

- The neural network consists of an input layer (4 neurons), a hidden layer (3 neurons), and an output layer (3 neurons).
- We use the sigmoid activation function for all neurons.
- The network is trained using backpropagation with stochastic gradient descent.
- The dataset is split into 80% training and 20% testing sets.

## 🔍 Code Explanation

1. **Data Loading**: The `read_dataset` function reads the Iris dataset from a CSV file.

2. **Neural Network Class**: The `NeuralNetwork` class encapsulates the network architecture and training process.

3. **Training**: The `train` method in the `NeuralNetwork` class implements the backpropagation algorithm.

4. **Evaluation**: The `evaluate_model` function calculates accuracy and generates a confusion matrix.

5. **User Interface**: We use ANSI color codes and emojis to create an engaging CLI interface.

6. **Main Function**: Orchestrates the entire process of loading data, training, and evaluation.

## 📚 Resources

To learn more about neural networks and implement your own, check out these resources:

1. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
2. [3Blue1Brown Neural Network Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
3. [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
4. [Deep Learning Book](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
5. [Neural Net implementation in C++](https://youtu.be/sK9AbJ4P8ao?si=ykOasvjrar82qcdp) by Dave Miller

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

Happy coding! 🎉 If you have any questions or suggestions, please open an issue or reach out to me on the socials.

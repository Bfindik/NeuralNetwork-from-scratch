# NeuralNetwork-from-scratch
In this project, I implemented a simple single and two-layer neural network and trained it on the IRIS dataset.
## Installation
- data_set= https://www.kaggle.com/datasets/uciml/iris <br/>
- in terminal: <br/>
   `pip install scikit-learn` <br/>
   `pip install numpy` <br/>
   `pip install pandas` <br/>
   `pip install matplotlib` <br/>(these codes can have different versions)
## Overview
- I try to code a simple neuron network with numpy and pandas libraries <br/>
- I encoded flower names as numbers:
  - 0 : Iris-setosa
  - 1 : Iris-versicolor
  - 2 : Iris-virginica

## Single Layer NeuralNetwork
![my](https://github.com/Bfindik/images/blob/main/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202023-03-10%20192712.png)
- Leaky ReLU activation function and Gradient Descent Algorithm implemented
- Error graphic created  for all training instances 
![graphic](https://github.com/Bfindik/images/blob/main/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202023-03-10%20192955.png)
## Two-Layer NeuralNetwork
- ReLU activation function and Gradient Descent Algorithm implemented
![my model](https://github.com/Bfindik/images/blob/main/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202023-03-07%20185953.png)
- In dataset,there are 150 sample.I used 80 percent of it for training and 20 percent for testing.<br/>
![result](https://github.com/Bfindik/images/blob/main/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202023-03-07%20184232.png)
![result2](https://github.com/Bfindik/images/blob/main/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202023-03-10%20190903.png)

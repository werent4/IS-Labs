'''
Implementation of Perceptron for 1st lab
'''

from typing import List, Tuple
from pathlib import Path
import numpy as np
import random

random.seed(42)

def create_dataset(data_path: Path) -> Tuple[List[List[float]], List[int]]:
    x1, x2, target = [], [], []
    with open(data_path) as file:
        for line in file:
            values = line.strip().split(',')
            x1.append(float(values[0]))
            x2.append(float(values[1]))
            target.append(float(values[2]))
    return [x1,x2], target


class Perceptron:
    def __init__(self,
                 features: List[List[float]],
                 target: List[int],
                 lr: float = 1.0) -> None:
        
        self.features = np.array(features) # Shape: (n_features, n_samples)
        self.target = np.array(target) # Shape: (n_samples,)

        n_features = self.features.shape[0]
        self.weights = np.random.uniform(0, 3, size=(n_features,))  # Shape: (n_features,)
        self.b = random.uniform(0,1)
        self.lr = lr
        
    def activation_fn(self, w_sum: List[float]) -> np.ndarray:  
        return np.where(w_sum >= 0, 1, -1)

    def backward(self, features: List[int], loss: List[float]) -> None:
        for i in range(len(self.weights)) :
            self.weights[i] += self.lr * np.dot(loss,features[i])

        self.b += self.lr * np.sum(loss)

    def loss(self, prediction) -> List[float]: # Shape: (n_features,)
        loss = self.target - prediction 
        return loss
    
    def forward(self, features: List[float] = []) -> List[float]: # Shape: (n_features,)
        if features == []:
            features = self.features
        else:
            features = np.array(features)
        w_sum = self.b + np.dot(features.T, self.weights)
        prediction = self.activation_fn(w_sum)

        return prediction
    
    def train(self, num_epochs: int = 50) -> None:

        for _ in range(num_epochs):
            prediction = self.forward()
            loss = self.loss(prediction)
            self.backward(self.features, loss)
            print(f"Epoch {_ + 1}, Loss: {np.sum(np.abs(loss))}")



if __name__ == "__main__":
    data_path = "Data.txt"
    features, target = create_dataset(data_path)
    print(features)

    test_size = int(0.2 * len(target))
    X_train = [feature[:-test_size] for feature in features]
    X_test = [feature[-test_size:] for feature in features]  

    y_train = target[:-test_size]
    y_test = target[-test_size:]

    p = Perceptron(X_train, y_train, lr= 0.01)

    p.train(num_epochs= 100)
    print("Trying to predict:", )
    print(p.forward(X_test))
    print("correct answers:\n",y_test)
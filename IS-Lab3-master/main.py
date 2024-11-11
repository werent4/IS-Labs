import numpy as np
import matplotlib.pyplot as plt

# RBF Network class
class RBFNetwork:
    def __init__(self, c1, r1, c2, r2):
        self.c1 = c1  # center of first RBF
        self.r1 = r1  # radius of first RBF
        self.c2 = c2  # center of second RBF
        self.r2 = r2  # radius of second RBF
        self.b = np.random.randn() * 0.1   # bias weight
        self.w1 = np.random.randn() * 0.1    # weight for first RBF
        self.w2 = np.random.randn() * 0.1   # weight for second RBF
        
    def rbf1(self, x):
        return np.exp(-(x - self.c1)**2 / (2 * self.r1**2))
    
    def rbf2(self, x):
        return np.exp(-(x - self.c2)**2 / (2 * self.r2**2))
    
    def forward(self, x):
        return self.b + self.w1 * self.rbf1(x) + self.w2 * self.rbf2(x)
    
    def train(self, X, y, learning_rate=0.01, epochs=1000):
        for _ in range(epochs):
            for i in range(len(X)):
                # Forward pass
                rbf1_output = self.rbf1(X[i])
                rbf2_output = self.rbf2(X[i])
                y_pred = self.forward(X[i])
                
                # Calculate error
                error = y[i] - y_pred
                
                # Update weights using perceptron learning rule
                self.b += learning_rate * error
                self.w1 += learning_rate * error * rbf1_output
                self.w2 += learning_rate * error * rbf2_output

#  Generate full dataset
x = np.arange(0.1, 1.01, (1-0.1)/40)
y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Split data into train and test sets (70% train, 30% test)
train_size = int(0.7 * len(x))
indices = np.random.permutation(len(x))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = x[train_indices], y[train_indices]
X_test, y_test = x[test_indices], y[test_indices]

# Initialize and train RBF network
rbf_net = RBFNetwork(c1=0.3, r1=0.3, c2=0.6, r2=0.3)
rbf_net.train(X_train, y_train, learning_rate=0.01, epochs=5000)

# Generate predictions for plotting
x_plot = np.linspace(0.1, 1, 200)
y_pred = np.array([rbf_net.forward(xi) for xi in x_plot])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, 'bo', label='Test data')
plt.plot(x_plot, y_pred, 'r-', label='RBF Network output')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF Network Approximation')
plt.legend()
plt.grid(True)
plt.show()

# Print final parameters
print(f"\nFinal parameters:")
print(f"b: {rbf_net.b:.4f}")
print(f"w1: {rbf_net.w1:.4f}")
print(f"w2: {rbf_net.w2:.4f}")
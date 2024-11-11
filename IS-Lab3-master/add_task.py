import numpy as np
import matplotlib.pyplot as plt

# RBF Network class
class RBFNetwork:
    def __init__(self, c1, r1, c2, r2):
        # Initialize centers and radii
        self.c1 = c1  # center of first RBF
        self.r1 = r1  # radius of first RBF
        self.c2 = c2  # center of second RBF
        self.r2 = r2  # radius of second RBF
        # Initialize weights and bias
        self.b = np.random.randn() * 0.1   # bias weight
        self.w1 = np.random.randn() * 0.1    # weight for first RBF
        self.w2 = np.random.randn() * 0.1   # weight for second RBF

    def rbf1(self, x):
        return np.exp(-((x - self.c1)**2) / (2 * self.r1**2))

    def rbf2(self, x):
        return np.exp(-((x - self.c2)**2) / (2 * self.r2**2))

    def forward(self, x):
        return self.b + self.w1 * self.rbf1(x) + self.w2 * self.rbf2(x)

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]
                
                # Forward pass
                rbf1_output = self.rbf1(xi)
                rbf2_output = self.rbf2(xi)
                y_pred = self.b + self.w1 * rbf1_output + self.w2 * rbf2_output

                # Calculate error
                error = yi - y_pred
                total_error += error**2

                # Gradients for weights and bias
                db = -2 * error
                dw1 = -2 * error * rbf1_output
                dw2 = -2 * error * rbf2_output

                # Gradients for centers and radii
                dc1 = -2 * error * self.w1 * rbf1_output * ((xi - self.c1) / (self.r1**2))
                dc2 = -2 * error * self.w2 * rbf2_output * ((xi - self.c2) / (self.r2**2))
                dr1 = -2 * error * self.w1 * rbf1_output * (((xi - self.c1)**2) / (self.r1**3))
                dr2 = -2 * error * self.w2 * rbf2_output * (((xi - self.c2)**2) / (self.r2**3))

                # Update parameters
                self.b -= learning_rate * db
                self.w1 -= learning_rate * dw1
                self.w2 -= learning_rate * dw2
                self.c1 -= learning_rate * dc1
                self.c2 -= learning_rate * dc2
                self.r1 -= learning_rate * dr1
                self.r2 -= learning_rate * dr2

            # Optional: Print the error every 500 epochs
            if (epoch+1) % 500 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Total Error: {total_error}")

# Generate full dataset
x = np.arange(0.1, 1.01, (1-0.1)/40)
y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Split data into train and test sets (70% train, 30% test)
np.random.seed(0)  # For reproducibility
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
plt.title('RBF Network Approximation with Trainable Centers and Radii')
plt.legend()
plt.grid(True)
plt.show()

# Print final parameters
print(f"\nFinal parameters:")
print(f"b: {rbf_net.b:.4f}")
print(f"w1: {rbf_net.w1:.4f}")
print(f"w2: {rbf_net.w2:.4f}")
print(f"c1: {rbf_net.c1:.4f}")
print(f"r1: {rbf_net.r1:.4f}")
print(f"c2: {rbf_net.c2:.4f}")
print(f"r2: {rbf_net.r2:.4f}")

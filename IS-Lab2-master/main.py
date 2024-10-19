import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_hidden = np.random.randn(input_size, hidden_size) * 0.1  # shape (input_size, hidden_size)
        self.b_hidden = np.zeros((1, hidden_size))  # shape (1, hidden_size)
        self.W_output = np.random.randn(hidden_size, output_size) * 0.1  # shape (hidden_size, output_size)
        self.b_output = np.zeros((1, output_size))  # shape (1, output_size)

    def activation_function(self, x, function_type='tanh'):
        if function_type == 'tanh':
            return np.tanh(x)
        elif function_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def linear_activation(self, x):
        # Linear activation function for the output layer
        return x

    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.W_hidden) + self.b_hidden  # hidden_input: shape (n_samples, hidden_size)
        self.hidden_output = self.activation_function(self.hidden_input, 'tanh')  # hidden_output: shape (n_samples, hidden_size)
        self.final_input = np.dot(self.hidden_output, self.W_output) + self.b_output  # final_input: shape (n_samples, output_size)
        output = self.linear_activation(self.final_input)  # output: shape (n_samples, output_size)
        return output

    def backpropagation(self, X, Y, output, learning_rate=0.01):
        # Backpropagation and weight updates
        output_error = Y - output  # output_error: shape (n_samples, output_size)
        output_delta = output_error  # output_delta: shape (n_samples, output_size) - Linear activation, gradient equals error

        hidden_error = np.dot(output_delta, self.W_output.T)  # hidden_error: shape (n_samples, hidden_size)
        hidden_delta = hidden_error * (1 - np.tanh(self.hidden_input) ** 2)  # hidden_delta: shape (n_samples, hidden_size)

        # Update weights and biases
        self.W_output += np.dot(self.hidden_output.T, output_delta) * learning_rate  # W_output: shape (hidden_size, output_size)
        self.b_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate  # b_output: shape (1, output_size)
        self.W_hidden += np.dot(X.T, hidden_delta) * learning_rate  # W_hidden: shape (input_size, hidden_size)
        self.b_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate  # b_hidden: shape (1, hidden_size)

    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        # Train the model for a number of epochs
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, Y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((Y - output) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')



if __name__ == "__main__":

    def create_surface_dataset(num_examples=20): # add task
        x1 = np.linspace(0, 1, int(np.sqrt(num_examples)))
        x2 = np.linspace(0, 1, int(np.sqrt(num_examples)))
        X1, X2 = np.meshgrid(x1, x2)
        X = np.c_[X1.ravel(), X2.ravel()]
        Y = (1 + 0.6 * np.sin(2 * np.pi * X[:, 0] / 0.7) + 0.3 * np.sin(2 * np.pi * X[:, 1])) / 2
        Y = Y.reshape(-1, 1)
        return X, Y

    def create_dataset(num_examples=20):
        X = np.linspace(0, 1, num_examples).reshape(-1, 1)
        
        Y = (1 + 0.6 * np.sin(2 * np.pi * X / 0.7) + 0.3 * np.sin(2 * np.pi * X)) / 2
        return X, Y

    X, y = create_surface_dataset()
    X_train = X[:-1]
    X_test = X[-1]
    y_train = y[:-1]
    y_test = y[-1]

    perceptron = MLP(input_size=2, hidden_size=8, output_size=1)

    print("Results before training:")
    predictions = perceptron.forward_pass(X_test)
    print(predictions)
    print("Actual answer: ", y_test)

    perceptron.train(X_train, y_train, epochs=1000, learning_rate=0.001)

    # Tests
    print("Results after training:")
    predictions = perceptron.forward_pass(X_test)
    print(predictions)
    print("Actual answer: ", y_test)

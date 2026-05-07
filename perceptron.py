import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights and bias randomly [0, 1]
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.eta = learning_rate # Learning rate

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.step_function(weighted_sum)

    def train(self, training_inputs, labels, epochs=100):
        """
        Iterates through the data and updates weights based on error.
        """
        for epoch in range(epochs):
            total_error = 0
            for x, target in zip(training_inputs, labels):
                prediction = self.predict(x)
                error = target - prediction
                
                if error != 0:
                    # Update weights: w = w + learning_rate * error * input
                    self.weights += self.eta * error * x
                    # Update bias: b = b + learning_rate * error
                    self.bias += self.eta * error
                    total_error += abs(error)
            
            # Stop if no mistakes were made in an epoch
            if total_error == 0:
                print(f"Converged at epoch {epoch}")
                break

# --- Training Example: Logical AND Gate ---
# Inputs are between 0 and 1
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1]) # Expected output for AND

# Initialize and Train
model = Perceptron(input_size=2)
print("Initial weights:", model.weights)

model.train(X, y)

# Final Testing
print("\nResults after training:")
for inputs in X:
    print(f"Input: {inputs} Prediction: {model.predict(inputs)}")
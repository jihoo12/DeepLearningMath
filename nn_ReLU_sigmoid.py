import numpy as np

# 1. Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # Gradient is 1 for x > 0, otherwise 0
    return (x > 0).astype(float)

# Note: We keep Sigmoid for the final output layer because 
# our targets are probabilities between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Data Setup
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 3. Hyperparameters
np.random.seed(42)
input_nodes = 2
hidden_nodes = 8  # Increased slightly to help ReLU converge on XOR
output_nodes = 1
learning_rate = 0.1 # Lowered slightly for stability with ReLU
epochs = 5000

# 4. Weight Initialization
# For ReLU, it's better to use small random numbers centered around 0
weights_input_hidden = np.random.randn(input_nodes, hidden_nodes) * 0.1
weights_hidden_output = np.random.randn(hidden_nodes, output_nodes) * 0.1
bias_hidden = np.zeros((1, hidden_nodes))
bias_output = np.zeros((1, output_nodes))

# 5. Training Loop
print("Training with ReLU...")
for epoch in range(epochs):
    # --- Forward Propagation ---
    # Hidden layer uses ReLU
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_activation)
    
    # Output layer uses Sigmoid (standard for binary classification)
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    # --- Backpropagation ---
    # Error at Output (Sigmoid)
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Error at Hidden Layer (ReLU)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_output)

    # --- Updating Weights and Biases ---
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(y - predicted_output))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# 6. Final Results
print("\nFinal Predictions:")
print(predicted_output)
print("\nRounded Predictions:")
print(np.round(predicted_output).astype(int))
import numpy as np

# 1. Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Data Setup
# Inputs (4 samples, 2 features)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Targets (4 samples, 1 output)
y = np.array([[0], [1], [1], [0]])

# 3. Hyperparameters
np.random.seed(42) # For consistent results
input_nodes = 2
hidden_nodes = 4
output_nodes = 1
learning_rate = 0.5
epochs = 10000

# 4. Weight Initialization
# Weights connect layers; Biases are added to the sums
weights_input_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
weights_hidden_output = np.random.uniform(size=(hidden_nodes, output_nodes))
bias_hidden = np.random.uniform(size=(1, hidden_nodes))
bias_output = np.random.uniform(size=(1, output_nodes))

# 5. Training Loop
print("Training...")
for epoch in range(epochs):
    # --- Forward Propagation ---
    # Input -> Hidden Layer
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    # Hidden -> Output Layer
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    # --- Backpropagation ---
    # Calculate error at Output
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Calculate error at Hidden Layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # --- Updating Weights and Biases (Gradient Descent) ---
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 2000 == 0:
        loss = np.mean(np.square(y - predicted_output))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# 6. Testing the Model
print("\nFinal Predictions after training:")
print(predicted_output)

# Rounding for clarity
print("\nRounded Predictions (0 or 1):")
print(np.round(predicted_output))
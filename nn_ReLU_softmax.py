import numpy as np

# 1. Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    # Subtracting np.max(x) for numerical stability (prevents overflow)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# 2. Data Setup
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# One-hot encoded targets: [Class 0, Class 1]
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) 

# 3. Hyperparameters
np.random.seed(42)
input_nodes = 2
hidden_nodes = 8
output_nodes = 2  # Changed from 1 to 2 for Softmax
learning_rate = 0.1
epochs = 220

# 4. Weight Initialization
weights_input_hidden = np.random.randn(input_nodes, hidden_nodes) * 0.1
weights_hidden_output = np.random.randn(hidden_nodes, output_nodes) * 0.1
bias_hidden = np.zeros((1, hidden_nodes))
bias_output = np.zeros((1, output_nodes))

# 5. Training Loop
for epoch in range(epochs):
    # --- Forward Propagation ---
    hidden_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = relu(hidden_activation)
    
    output_activation = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = softmax(output_activation)

    # --- Backpropagation ---
    # For Softmax with Mean Squared Error (or Cross-Entropy), 
    # the error gradient simplifies to (predicted - target)
    error = y - predicted_output
    
    # Gradient for Softmax output
    d_predicted_output = error / X.shape[0] 
    
    # Error at Hidden Layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_output)

    # --- Update Weights ---
    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(y - predicted_output))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# 6. Final Results
print("\nFinal Probabilities (Class 0 vs Class 1):")
print(predicted_output)

print("\nFinal Class Predictions:")
print(np.argmax(predicted_output, axis=1)) # Returns the index of the highest prob
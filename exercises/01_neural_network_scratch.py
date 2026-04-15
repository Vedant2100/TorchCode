import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
EXERCISE: Implement a simple Neural Network from scratch using PyTorch.
This exercise will guide you through creating a multi-layer perceptron (MLP) 
for classification.

Complete the TODOs below.
"""

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO 1: Initialize the weights and biases for two linear layers.
        # Layer 1: input_dim -> hidden_dim
        # Layer 2: hidden_dim -> output_dim
        # Hint: Use nn.Parameter and torch.randn or torch.empty
        
        self.w1 = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b1 = nn.Parameter(torch.empty(hidden_dim))
        self.w2 = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(output_dim))

        self.reset_parameters()


    def reset_parameters(self):
        # TODO 2: Initialize parameters using Kaiming He initialization for weights 
        # and zeros for biases.
        # Hint: std = 1 / sqrt(input_dim)
        with torch.no_grad():
            self.w1.data.normal_(0, math.sqrt(2. / self.w1.size(1)))
            self.b1.data.zero_()
            self.w2.data.normal_(0, math.sqrt(2. / self.w2.size(1)))
            self.b2.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3: Implement the forward pass.
        # 1. Linear transformation 1: x @ W1.T + b1
        # 2. Activation function: ReLU
        # 3. Linear transformation 2: h @ W2.T + b2
        # 4. Final Activation: Softmax (dim=-1)
        z1 = F.linear(x, self.w1, self.b1)
        a1 = F.relu(z1)
        z2 = F.linear(a1, self.w2, self.b2)
        out = F.softmax(z2, dim=-1)
        
        return out

def train_step(model, optimizer, x, y):
    # TODO 4: Implement a single training step.
    # 1. Zero gradients
    # 2. Forward pass
    # 3. Compute Negative Log Likelihood (NLL) Loss
    # 4. Backward pass
    # 5. Optimizer step
    
    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(torch.log(output), y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    # Test your implementation
    torch.manual_seed(42)
    
    input_dim, hidden_dim, output_dim = 10, 20, 3
    model = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)
        
    # Check parameters
    assert model.w1 is not None, "Weight1 not initialized"
    assert model.b1 is not None, "Bias1 not initialized"
    assert model.w2 is not None, "Weight2 not initialized"
    assert model.b2 is not None, "Bias2 not initialized"
    print("✅ Model parameters initialized.")

    # Check forward pass
    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5, output_dim), f"Expected shape (5, {output_dim}), got {output.shape}"
    assert torch.allclose(output.sum(dim=-1), torch.ones(5)), "Outputs should sum to 1 (Softmax)"
    print("✅ Forward pass successful.")

    # Check training step
    y = torch.randint(0, output_dim, (5,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_step(model, optimizer, x, y)
    print("✅ Training step completed.")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import render
import matplotlib.pyplot as plt

# Generate synthetic data points on the sphere
def generate_data(num_points):
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    z = np.random.uniform(-1, 1, num_points)

    data = np.column_stack((x, y, z))
    labels = np.linalg.norm(data, axis=1) - 1

    return data.astype(np.float32), labels.astype(np.float32)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Generate training data
num_train_points = 1000
train_data, train_labels = generate_data(num_train_points)

# Convert to PyTorch tensors
train_data = torch.from_numpy(train_data)
train_labels = torch.from_numpy(train_labels)
# train_labels -= 0.5
# train_labels *= 2

# Set up the model, loss function, and optimizer
input_size = 3  # Dimensionality of the input data (x, y, z coordinates)
hidden_size = 32  # Number of hidden units in each layer
output_size = 1  # Binary classification (inside or outside the sphere)

model = MLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000

def eikonal_loss(gradient):
    # Compute the L2 norm of the gradient
    gradient_norm = torch.norm(gradient, p=2, dim=1)

    # Enforce the Eikonal equation by penalizing the squared difference from 1
    eikonal_penalty = torch.mean((gradient_norm - 1.0) ** 2)

    return eikonal_penalty

for epoch in range(num_epochs):
    train_data.requires_grad = True
    model.train()


    outputs = model(train_data)
    loss = criterion(outputs.view(-1), train_labels)

    # Compute the gradient of the output w.r.t. the input
    input_data = train_data.clone().detach().requires_grad_(True)
    output_grad = torch.autograd.grad(outputs=outputs, inputs=input_data,
                                      grad_outputs=torch.ones_like(outputs),
                                      create_graph=True, retain_graph=True,
                                      only_inputs=True, allow_unused=True)
    print(output_grad)
    eikonal_penalty = eikonal_loss(output_grad)
    total_loss = loss + eikonal_penalty

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training finished!")


# Test the model on new data points (data points on the sphere)
num_test_points = 100
test_data, test_labels = generate_data(num_test_points)

with torch.no_grad():
    model.eval()
    test_outputs = model(torch.from_numpy(test_data))

predicted_labels = (test_outputs >= 0.5).squeeze().numpy().astype(int)
correct = np.sum(predicted_labels == test_labels.astype(int))
accuracy = correct / num_test_points * 100
print(f"Test Accuracy: {accuracy:.2f}%")


# Now render the learned levelset
def levelset(x, y, z):
    p = torch.stack([x, y, z], dim=-1)
    r = model(p).flatten()
    return r

image = render.render_image(levelset, resolution=(100, 100), max_distance=10.0, light_direction=torch.tensor([-1.0, -1.0, 1.0]))

plt.imshow(image)
plt.axis('off')
plt.show()

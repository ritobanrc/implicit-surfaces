import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import render
import matplotlib.pyplot as plt

device = render.device

## THIS CODE IS BASED ON https://github.com/ishit/nie/blob/main/models.py
# Neural Implicit Evolution
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self, const=30.):
        super().__init__()
        self.const = const

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.const * input)


class Net(nn.Module):
    """ Decoder conditioned by adding.

    Example configuration:
        hidden_size: 256
        n_blocks: 5
        out_dim: 3  # we are outputting the gradient
        sigma_condition: True
        xyz_condition: True
    """
    def __init__(self, input_size, output_size, hidden_size, n_blocks):
        super().__init__()
        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(input_size, hidden_size))
        for _ in range(n_blocks):
            self.blocks.append(nn.Linear(hidden_size, hidden_size))
        self.blocks.append(nn.Linear(hidden_size, output_size))
        # self.act = Sine()
        self.act = nn.GELU()

        # Initialization
        self.apply(sine_init)
        self.blocks[0].apply(first_layer_sine_init)

    def forward(self, x):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        net = x  # (bs, n_points, dim)
        for block in self.blocks[:-1]:
            net = self.act(block(net))
        out = self.blocks[-1](net)
        return out


# Generate synthetic data points on the sphere
def generate_data(num_points):
    x = torch.from_numpy(np.random.uniform(-1, 1, num_points)).to(device)
    y = torch.from_numpy(np.random.uniform(-1, 1, num_points)).to(device)
    z = torch.from_numpy(np.random.uniform(-1, 1, num_points)).to(device)

    data = torch.stack([x, y, z])
    # labels = np.linalg.norm(data, axis=1) - 1
    labels = render.default_levelset(x, y, z)

    return data, labels

# Generate training data
num_train_points = 1000
train_data, train_labels = generate_data(num_train_points)

# Set up the model, loss function, and optimizer
input_size = 3  # Dimensionality of the input data (x, y, z coordinates)
hidden_size = 32  # Number of hidden units in each layer
output_size = 1  # Binary classification (inside or outside the sphere)
n_blocks = 4

# model = MLP(input_size, hidden_size, output_size).to(device)
model = Net(input_size, output_size, hidden_size, n_blocks).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000

# def eikonal_loss(gradient):
    # # Compute the L2 norm of the gradient
    # gradient_norm = torch.norm(gradient, p=2, dim=1)

    # # Enforce the Eikonal equation by penalizing the squared difference from 1
    # eikonal_penalty = torch.mean((gradient_norm - 1.0) ** 2)

    # return eikonal_penalty

for epoch in range(num_epochs):
    train_data.requires_grad = True
    model.train()


    outputs = model(train_data)
    loss = criterion(outputs.view(-1), train_labels)

    # Compute the gradient of the output w.r.t. the input
    input_data = train_data.clone().detach().requires_grad_(True)
    # output_grad = torch.autograd.grad(outputs=outputs, inputs=input_data,
                                      # grad_outputs=torch.ones_like(outputs),
                                      # create_graph=True, retain_graph=True,
                                      # only_inputs=True, allow_unused=True)
    # print(output_grad)
    # eikonal_penalty = eikonal_loss(output_grad)
    total_loss = loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}")

print("Training finished!")

# Now render the learned levelset
def levelset(x, y, z):
    p = torch.stack([x, y, z], dim=-1)
    r = model(p).flatten()
    return r

image = render.render_image(levelset, resolution=(200, 200), max_distance=10.0, light_direction=torch.tensor([-1.0, -1.0, 1.0]))

plt.imshow(image)
plt.axis('off')
plt.show()

import torch
import matplotlib.pyplot as plt
import time

# Define the levelset function (same as before)
# def levelset(x, y, z):
    # return torch.sqrt(x**2 + y**2 + z**2) - 1.0
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = 'cpu'
print("Using device: ", device)

def union_levelset(levelset_func1, levelset_func2):
    def union(x, y, z):
        return torch.min(levelset_func1(x, y, z), levelset_func2(x, y, z))
    return union

def sphere(radius):
    return lambda x, y, z: torch.sqrt(x ** 2 + y ** 2 + z ** 2) - radius

def box(bounds):
    def box_sdf(x, y, z):
        p = torch.stack([x, y, z], dim=-1)
        q = torch.abs(p) - bounds
        return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.min(torch.max(q[:, 0], torch.max(q[:, 1], q[:, 2])), torch.tensor(0.0)) - 0.001

    return box_sdf

def torus(s, t):
    return lambda x, y, z: torch.sqrt((torch.sqrt(x**2 + y**2) - s)**2 + z**2) - t

def translate(func, dx, dy, dz):
    return lambda x, y, z: func(x - dx, y - dy, z - dz)


def default_levelset(x, y, z):
    box1 = translate(box(torch.tensor([0.25, 0.25, 0.25], device=device)), 0, -1, 0)
    sphere2 = sphere(0.01*(torch.sin(50*y) + 1) + 0.25)
    torus3 = translate(torus(0.2, 0.1), 0, 1, 0)
    return union_levelset(union_levelset(box1, sphere2), torus3)(x, y, z)

# Calculate the normal vector using autodifferentiation
def calculate_normal(levelset, pos):
    if pos.is_leaf:
        pos.requires_grad = True
    distance = levelset(pos[:, 0], pos[:, 1], pos[:, 2])
    normal = torch.autograd.grad(distance, pos, torch.ones_like(distance), create_graph=True)[0]
    return normal

# Ray marching function (updated to handle batches of rays)
def ray_march(levelset, origin, directions, max_steps=100, max_distance=100.0, epsilon=1e-6):
    t = torch.zeros(directions.shape[0], device=device)  # Initialize t for each ray to 0.0
    for i in range(max_steps):
        pos = origin + t[:, None] * directions
        distance = levelset(pos[:, 0], pos[:, 1], pos[:, 2])
        mask = torch.logical_or(torch.abs(distance) < epsilon, torch.abs(distance) > max_distance)
        t[~mask] += distance[~mask]
        if torch.all(t > max_distance):
            break
    t[t > max_distance] = max_distance
    return t

# Lambertian diffuse shading function
def lambertian_shading(normals, light_direction):
    light_direction = light_direction.reshape(1, 1, -1)
    light_direction /= torch.norm(light_direction, dim=-1, keepdim=True)
    dot_product = torch.sum(normals * light_direction, dim=-1)
    dot_product = torch.clamp(dot_product, min=0.0)  # Ensure the dot product is non-negative (max with 0.0)
    return dot_product

# Updated rendering function with Lambertian shading
def render_image(levelset, resolution=(200, 200), max_distance=100.0, light_direction=torch.tensor([1.0, 1.0, -1.0])):
    height, width = resolution
    # image = 1/255 * torch.ones((width, height, 3)) * torch.tensor([21, 21, 21], dtype=torch.float32)  # Background color (dark grey)
    image = torch.zeros((width, height, 3), dtype=torch.float32, device=device)
    fov = torch.tensor([1.0, 1.0])
    aspect_ratio = width / height
    
    x = torch.linspace(-fov[0] / 2, fov[0] / 2, width, device=device)
    y = torch.linspace(-fov[1] / 2, fov[1] / 2, height, device=device)
    
    # Create a grid of pixel positions
    x, y = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate ray directions for all pixels using broadcasting
    direction = torch.stack([x * aspect_ratio, y, -torch.ones_like(x)], dim=-1)
    direction /= torch.norm(direction, dim=-1, keepdim=True)
    
    # Batched ray marching
    origin = torch.tensor([0.0, 0.0, 3.0], device=device)
    t = ray_march(levelset, origin, direction.reshape(-1, 3), max_distance=max_distance)
    
    # Reshape t back to the image size
    t = t.reshape(width, height)
    
    # Calculate intersection points
    intersection_points = origin + t[:, :, None] * direction
    
    # Calculate surface normals
    normals = calculate_normal(levelset, intersection_points.reshape(-1, 3))
    normals = normals.reshape(width, height, 3)
    
    # Lambertian shading
    shading = lambertian_shading(normals, light_direction.to(device))

    ambient = torch.tensor([0.3, 0.4, 0.5], device=device)
    image[t >= max_distance] = ambient

    albedo = torch.tensor([0.2, 0.5, 0.2], dtype=torch.float32, device=device)  # Green
    image[t < max_distance] = ambient * albedo

    color = albedo * shading.unsqueeze(-1)  # Apply shading to the object color
    image += color
    # image = torch.clamp(image, 0, 255).byte()
    
    # Set the shading as the image color
    image = image.cpu().detach().numpy()  # Detach the tensor before converting to NumPy array
    
    return image

def main():
    # Render the image with max_distance=10.0 and light direction=[1.0, 1.0, -1.0]
    start_time = time.time() 
    image = render_image(default_levelset, resolution=(1280, 720), max_distance=100.0, light_direction=torch.tensor([-1.0, -1.0, 1.0]))
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"Rendering time: {elapsed_time:.4f} seconds")

    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

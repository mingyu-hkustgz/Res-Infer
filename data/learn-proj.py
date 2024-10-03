import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from utils import fvecs_read, fvecs_write, ivecs_read, ivecs_write
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class LinearProjection(nn.Module):
    def __init__(self, input_dim, projected_dim):
        super(LinearProjection, self).__init__()
        # Linear projection without bias
        self.projection = nn.Linear(input_dim, projected_dim, bias=False)
        self.projection.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.projection(x)


def projection_loss(original_vectors, projected_vectors, temperature=1.0):
    # Compute dot products of original vectors
    original_dot_products = torch.matmul(original_vectors, original_vectors.t())

    # Compute dot products of projected vectors
    projected_dot_products = torch.matmul(projected_vectors, projected_vectors.t())

    # Calculate absolute differences between dot products
    differences = torch.abs(original_dot_products - projected_dot_products)

    # Use softmax to approximate the maximum value
    # Temperature controls the "sharpness" of the softmax distribution
    softmax_differences = torch.softmax(differences / temperature, dim=-1)

    # Compute weighted average error
    loss = torch.sum(softmax_differences * differences)

    return loss



# Set parameters
input_dim = 100
projected_dim = 50
num_vectors = 1000
learning_rate = 0.001
num_epochs = 3
initial_temperature = 1.0
temperature_increase_factor = 0.99  # Factor to increase temperature each epoch
batch_size = 2048
source = '/home/yming/DATA/vector_data'
device = torch.device("cuda:1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    print(f"Learn Proj- {dataset}")
    # path
    path = os.path.join(source, dataset)
    base_path = os.path.join(path, f'{dataset}_base.fvecs')
    query_path = os.path.join(path, f'{dataset}_query.fvecs')
    query = fvecs_read(query_path)
    # read data vectors
    original_vectors = fvecs_read(base_path)
    original_vectors = torch.from_numpy(original_vectors).to(device)
    query_vectors = fvecs_read(query_path)
    query_vectors = torch.from_numpy(query_vectors).to(device)

    VecData = torch.utils.data.TensorDataset(original_vectors)
    origin_vector_loader = torch.utils.data.DataLoader(VecData, batch_size=batch_size, shuffle=True)
    N, D = original_vectors.shape
    input_dim = D
    projected_dim = 64

    # Initialize the model
    model = LinearProjection(input_dim, projected_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    temperature = initial_temperature
    for epoch in tqdm(range(num_epochs)):
        for vecs in origin_vector_loader:
            vecs = vecs[0].to(device)
            optimizer.zero_grad()
            # Forward pass
            projected_vectors = model(vecs)
            # Compute loss
            loss = projection_loss(vecs, projected_vectors, temperature)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Temperature: {temperature:.4f}')
            temperature *= temperature_increase_factor

    # Project vectors using the trained model
    with torch.no_grad():
        final_projected_vectors = model(original_vectors).cpu().numpy()
        final_query_vectors = model(query_vectors).cpu().numpy()
        dot_product = (final_projected_vectors[:] * final_query_vectors[0]).sum(axis=0)
        print(f"Learn proj dimension: {projected_dim}")
        plt.hist(dot_product, bins=100)
        plt.savefig(f"./figure/{dataset}-learn-proj.png")
        plt.show()

    print("Training completed.")

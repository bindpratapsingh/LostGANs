import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Get GPU name
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Verify device
x = torch.rand(5, 5).cuda()  # Move tensor to GPU
print("Tensor on GPU:", x)

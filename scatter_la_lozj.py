import torch
tmp_x = torch.zeros(3, 5)  # A 3x5 matrix of zeros
max_ids = torch.tensor([[1, 3], [0, 2], [4, 1]])  # Indices to scatter values
max_values = torch.tensor([[0.9, 0.8], [0.7, 0.6], [1.0, 0.5]])  # Values to scatter
print(tmp_x)
# Scatter values into tmp_x at specified indices
tmp_x.scatter_(1, max_ids, max_values)

print(tmp_x)
import torch
import torch.nn as nn
from models.MMR import MMR
from models.MMR import MMR_losses

# Define dummy data parameters
batch_size = 8  # Adjust as needed
seq_len = 10  # Length of the token sequence, adjust as needed
hidden_dim = 1024  # Size of the feature vectors
num_heads = 4  # Number of attention heads
ITEM_lyrs = 2  # Number of ITEM layers
MTD_lyrs = 2  # Number of MTD layers
projection_dim = 512  # Projection dimension

# Create dummy data (projected recipe and image embeddings)
recipe_tokens = torch.randn(batch_size, seq_len, hidden_dim)  # Shape: [batch_size, seq_len, hidden_dim]
image_tokens = torch.randn(batch_size, seq_len, hidden_dim)  # Shape: [batch_size, seq_len, hidden_dim]

# Create labels for testing
labels = torch.randint(0, 2, (batch_size,)).float()  # Random binary labels for instance loss calculation

# Instantiate the MMR model
mmr_model = MMR(hidden_dim=hidden_dim, num_heads=num_heads, ITEM_lyrs=ITEM_lyrs, MTD_lyrs=MTD_lyrs, projection_dim=projection_dim)

# Forward pass through the model
logits = mmr_model(recipe_tokens, image_tokens, labels)

sem_loss = instance_semantic_loss(image_tokens, recipe_tokens, labels, 1, mode='semantic')
inst_loss = instance_semantic_loss(image_tokens, recipe_tokens, labels, 1, mode='instance')
itm_loss = itm_loss(mmr_logits, labels)

loss = MMR_losses.total_loss(self, labels, image_tokens, text_tokens, mmr_logits, margin=1.0, instance_weight=1.0, sem_weight=1.0, itm_weight=1.0)

# Print the outputs for verification
print("Logits:", logits)
print("Instance Loss:", instance_loss.item())
print("Semantic Loss:", semantic_loss.item())

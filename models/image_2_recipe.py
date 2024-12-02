import torch
import torch.nn as nn
import torch.functional as F

class Image2Recipe(nn.Module):
    def __init__(self, image_encoder, recipe_encoder, projection_dim=512): ##Im not sure projection_dim we should put
        super().__init__()
        
        self.image_encoder = image_encoder
        self.recipe_encoder = recipe_encoder
        self.mmr = mmr
        
        # Projection layers to map both modalities to a shared embedding space
        self.image_projection = nn.Linear(image_encoder.image_embedding_dim, projection_dim)
        self.recipe_projection = nn.Linear(recipe_encoder.output_size, projection_dim)
        
        self.normalize = nn.functional.normalize

    def forward(self, images, image_labels, src):
        # Get encodings from both encoders
        image_logits, image_encodings, text_features = self.image_encoder(images, image_labels)
        t_R, e_R = self.recipe_encoder(src)
        
        ####Not completely sure what should be happening below but I know they should probably be in same dimension
        image_embeddings_proj = self.image_projection(image_encodings)
        recipe_embeddings_proj = self.recipe_projection(e_R)
        
        # Normalize the embeddings
        image_embeddings_proj = self.normalize(image_embeddings_proj, p=2, dim=-1)
        recipe_embeddings_proj = self.normalize(recipe_embeddings_proj, p=2, dim=-1)

        output_logits = self.mmr(recipe_embeddings_proj, image_embeddings_proj)
        
        # Return logits and normalized embeddings for similarity
        return {
            "output_logits": output_logits,
            "image_logits": image_logits,
            "image_embeddings": image_embeddings_proj,
            "recipe_embeddings": recipe_embeddings_proj
        }

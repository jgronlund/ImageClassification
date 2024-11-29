# CITATIONS: basis for MMR section is https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9857479, and A4
# The entirety of this script is based heavily on these

# Imports
import numpy as np
import torch
from torch import nn
import random


# Goal: retrieve the recipe corresponding to a dish image and vice versa
# We feed the info from the image and recipers into: "the two sequences into the MultiModal Regularization (MMR) module, 
# which computes a fine-grained alignement score (Image-Text Matching loss) between the input image and recipe..
# MMR consists of:
#     - transformer decoders
#     - Image-Text Matching loss
#     - multimodal modules are used only during training
#     - Replace simple projection layer is added on top of the feature representation that computes the contrastive loss with a more complicated multimodal
#       module, and the contrastive loss is replaced by ITM.

# The main block: 
# - transformer decoder which particularly uses cross attention: the Queries (Q) come from one modality while the Keys (K) and Values
#   (V) come from the other modality (we do not distinguish between K and V, which denoted as KV). --> does it matter which way?
#   each vector of Q is updated by taking a weighted sum of the vectors of KV of the other modality, 
#   where each weight is the similarity of each one of these KV vectors to the underlying Q vector. 
# - Inter-connected transformers: 
#     - Image Tokens Enhancement Module (ITEM) to enhance the image tokens
#     - Multimodal transformer decoder (MTD)

class MMR_losses(nn.Module):
    def __init__(self, margin=0.3, triplet_weight=1.0, match_weight=1.0, sem_weight=1.0):
        super().__init__()
        self.margin = margin  # Distance btwn img and sample < img to rand by margin! CITATION: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9857479
        self.triplet_weight = triplet_weight
        self.match_weight = match_weight
        self.sem_weight = sem_weight
        self.itm_loss = nn.BCELoss()  # CITATION: https://github.com/mshukor/TFood/blob/main/bootstrap.pytorch/bootstrap/templates/default/project/models/criterions/criterion.py

    def itm_loss(self, alignment_score, labels):
        # Litm = −EtR,tI∼D[y log(s(tR, tI ))+ (1 − y) log(1 − s(tR, tI ))]
        return self.itm_loss(alignment_score, labels)

    def triplet_loss(self, img_anc, positive, negative):
        # l(xa, xp, xn, α)=[d(xa, xp (positive)) + α − d(xa, xn (negative))]+
        d_ap = torch.norm(img_anc - positive, p=2, dim=-1)
        d_an = torch.norm(img_anc - negative, p=2, dim=-1)
        return torch.relu(d_ap + self.margin - d_an).mean()

  # "The semantic loss Lsem is the same as the instance loss except for the selection of positive and negative samples"
  def sem_loss(self, img_anc, positive, negative):
        # l(xa, xp, xn, α)=[d(xa, xp (positive)) + α − d(xa, xn (negative))]+
        d_ap = torch.norm(img_anc - positive, p=2, dim=-1)
        d_an = torch.norm(img_anc - negative, p=2, dim=-1)
        return torch.relu(d_ap + self.margin - d_an).mean()

    def total_loss(self, itm_score, labels, img_anc, positive, negative):
        # L = Litc + λsemLsem + λitmLitm -->  
        loss_itm = self.itm_loss(itm_score, labels)
        loss_triplet = self.triplet_loss(img_anc, positive, negative)
        return self.triplet_weight * loss_triplet + self.match_weight * loss_itm + self.sem_weight * loss_sem


class MMR(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=2, num_layers=2, projection_dim=512):
        super().__init__()

        # Start the class. What info do I have? The image (encoded) and the recipe (encoded) as well as user determined 
        # variation.

        # Image Tokens Enhancement Module (ITEM) to enhance the image tokens
        self.decoder_ITEM = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.hidden_dim, self.num_heads, batch_first=True), num_layers=num_layers)
        # Multimodal transformer decoder (MTD)
        self.decoder_MTD = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.hidden_dim, self.num_heads, batch_first=True), num_layers=num_layers)

        # Projec: - Replace simple projection layer is added on top of the feature representation that computes the contrastive loss with a more complicated multimodal
        # module, and the contrastive loss is replaced by ITM.
        self.projection = nn.Linear(hidden_dim, projection_dim)
        # the second part!! matching section --> IT FOLLOWS the other section, so it should be sequential!
        self.match_score = nn.Sequential(nn.Linear(projection_dim, 1), nn.Sigmoid())
        
      
    def alignment_score(self, image_tokens, recipe_tokens):
        # Image Tokens Enhancement Module (ITEM) to enhance the image tokens
        enhanced_image_tokens = self.decoder_ITEM(image_tokens, recipe_tokens)

        # Multimodal

        # Project to shared space --> linear layer

        # match 

        return score



    def forward(self, image_tokens, recipe_tokens):
      score = self.alignment_score(image_tokens, recipe_tokens)
      


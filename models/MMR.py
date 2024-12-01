# CITATIONS: basis for MMR section is https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9857479, A4, 
# https://github.com/mshukor/TFood/blob/main/recipe1m/visu/modality_to_modality_top5.py#L120 ,
# https://github.com/mshukor/TFood/blob/main/recipe1m/models/networks/trijoint.py#L22
# The entirety of this script is based heavily on these

# Imports
import numpy as np
import torch
from torch import nn
import random
# Reuse Chelsea's code
# from models.reciper_encoder import TransformerEncoder TransformerDecoder https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html

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
    def __init__(self, triplet_weight=0.0, match_weight=1.0, sem_weight=0.0):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.match_weight = match_weight
        self.sem_weight = sem_weight
        self.itm_loss = nn.BCELoss()  # CITATION: https://github.com/mshukor/TFood/blob/main/bootstrap.pytorch/bootstrap/templates/default/project/models/criterions/criterion.py

    def itm_loss(self, alignment_score, labels):
        # Litm = −EtR,tI∼D[y log(s(tR, tI ))+ (1 − y) log(1 − s(tR, tI ))] --> it is just the BCE loss
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

class TDB(nn.Module):
    '''
    Self attention,
    cross attention, 
    feed forward and layer normalization layers with
    residual connections, which are repeated N times!
    '''
    def __init__(self, hidden_dim, heads, hidden_factor=4):
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Self attention --> ouput is attn_output, attn_output_weights, so we only care about the first
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)
        self.norm_sa = nn.LayerNorm(hidden_dim)
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)
        self.norm_ca = nn.LayerNorm(hidden_dim)

        # Feed forward layers --> expand and compress to keep shape --> Sequential to only call one thing... ReLU for nonlinearity but they use sigmoid
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * hidden_factor), nn.ReLU(), nn.Linear(hidden_dim * hidden_factor, hidden_dim))
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, src):
        # Input: query, key, value
        tgt = self.self_attn(tgt, tgt, tgt)[0]
        tgt = self.norm_sa(tgt)

        # We don't differentiate between key and value per the paper
        tgt = self.cross_attn(tgt, src, src)[0]
        tgt = self.norm_ca(tgt)

        tgt = self.feed_forward(tgt)
        tgt = self.norm_ff(tgt)

        return tgt


class MMR(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=4, ITEM_lyrs=1, MTD_lyrs=4, projection_dim=512):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ITEM_lyrs = ITEM_lyrs
        self.MTD_lyrs = MTD_lyrs
        self.projection_dim = projection_dim

        # Start the class. What info do I have? 
        #       The image (encoded) --> Image encoder returns image_logits, image_encodings, text_features
        #       The recipe (encoded) --> 
        #       User inputs --> heads, loss weights

        # What do I want to do? The Transformer Decoders Block (TDB) -> ITEM -> TDB ...
        # ITEM: The image tokens are enhanced by the ITEM module by attending to the recipe tokens. The recipe tokens are fed to MTD as Q and the enhanced
        # image tokens as K and V. After fusing the two modalities an ITM
        # loss is applied. Note that this module is used only during training (phase)

        # easier one! Image Tokens Enhancement Module (ITEM) to enhance the image tokens 
        # CITATION: https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html 
        self.ITEM = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.hidden_dim, self.num_heads, batch_first=True), num_layers=self.ITEM_lyrs)
        # Multimodal transformer decoder (MTD) --> using tdb implentation
        self.MTD = nn.ModuleList([TDB(self.hidden_dim, self.num_heads) for _ in range(self.MTD_lyrs)])

        # Project: - Replace simple projection layer is added on top of the feature representation that computes the contrastive loss with a more complicated multimodal
        # module, and the contrastive loss is replaced by ITM. Each must be projected separately!
        self.image_proj = nn.Linear(self.hidden_dim, self.projection_dim)
        self.recipe_proj = nn.Linear(self.hidden_dim, self.projection_dim)
        
        # the second part!! matching section --> IT FOLLOWS the other section, so it should be sequential!
        self.match_score = nn.Sequential(nn.Linear(projection_dim, 1), nn.Sigmoid())
        
    def forward(self, recipe_tokens, image_tokens):

        # ITEM: Enhance image tokens by focusing on recipe tokens
        enhanced_image_tokens = image_tokens.transpose(0, 1)
        recipe_focus = recipe_tokens.transpose(0, 1)
        enhanced_image_tokens = self.ITEM(tgt=enhanced_image_tokens, memory=recipe_focus).transpose(0, 1)

        # MTD: The recipe tokens are fed to MTD as Q and the enhanced image tokens as K and V. Then the modalities are fused
        fused_tokens = recipe_tokens.transpose(0, 1)
        for layer in self.MTD_layers:
            fused_tokens = layer(fused_tokens, enhanced_image_tokens)

        # Project the two embeddings into space for ITM loss
        recipe_projection = self.recipe_proj(fused_tokens.transpose(0, 1)[:, 0, :]) 
        image_projection = self.image_proj(enhanced_image_tokens.transpose(0, 1)[:, 0, :]) 
        concat_projections = torch.cat(recipe_projection, image_projection)
        
        # After fusing the two modalities an ITM loss is applied --> not the triplet loss?
        logits = self.match_score(torch.sigmoid(concat_projections))

        return logits
        
    def compute_loss(self, tgt_labels, logits, loss_type='ITM', triplet_weight=0.0, match_weight=1.0, sem_weight=0.0):
        """
        Calculate the losses--> input target and the logits from the forward pass
        """
        # if loss_type == 'ITM':
        itm_loss = MMR_losses.itm_loss(torch.sigmoid(logits), tgt_labels.float())
        return itm_loss

    def seed_torch(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
      


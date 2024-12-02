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
    def __init__(self, margin=1.0, instance_weight=1.0, sem_weight=1.0, itm_weight=1.0):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.match_weight = match_weight
        self.sem_weight = sem_weight
        self.itm_loss = nn.BCELoss()  # CITATION: https://github.com/mshukor/TFood/blob/main/bootstrap.pytorch/bootstrap/templates/default/project/models/criterions/criterion.py

    def fast_distance(A,B):
        # SOURCE: COPIED FROM https://github.com/mshukor/TFood/
        # A and B must have norm 1 for this to work for the ranking
        A = torch.nn.functional.normalize(A, p=2, dim=-1)
        B = torch.nn.functional.normalize(B, p=2, dim=-1)
        return torch.mm(A,B.t()) * -1
    
    def itm_loss(self, mmr_out, labels):
        # Litm = −EtR,tI∼D[y log(s(tR, tI ))+ (1 − y) log(1 − s(tR, tI ))] --> it is just the BCE loss
        return self.itm_loss(mmr_out, labels)
        
    # Measures similarity between an anchor and its directly associated positive while ensuring dissimilarity from negatives in the same batch.
    # "The semantic loss Lsem is the same as the instance loss except for the selection of positive and negative samples"
    # Encourages embeddings to reflect class-level semantics, ensuring that embeddings from the same class are closer, even if they are not direct pairs.
    def instance_semantic_loss(self, img_embeddings, txt_embeddings, labels, margin=1.0, mode='instance'):
        '''
        mode is instance or semantic,
        image embadggings of shape batch, hidden
        text embeddings of size batch, hidden, 
        labels of size batch, 1
        '''
        # l(xa, xp, xn, α)=[d(xa, xp (positive)) + α − d(xa, xn (negative))]+
        batch_size = distances.size()[0]
        
        if mode == 'instance':
            # Here we need to check that the images are getting the right text and vice versa
            distances = self.fast_distance(image_embeds, text_embeds)
            positives = torch.arange(batch_size, device=image_embeds.device)
            loss_im_txt = F.cross_entropy(-distances, positives)
            loss_txt_im = F.cross_entropy(-distances.T, positives)
            
            return (loss_im_txt + loss_txt_im).mean()
            
        if mode == 'semantic':
            # Get the same class/value
            labels = labels.unsqueeze(1)
            positives_mask = (labels == labels.T) 
            negatives_mask = ~positives_mask
    
            txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=-1)
            txt_distances = fast_distance(txt_embeddings, txt_embeddings)

            img_embeddings = torch.nn.functional.normalize(img_embeddings, p=2, dim=-1)
            img_distances = fast_distance(img_embeddings, img_embeddings)
            
            positive_txt_distances = text_distances * positives_mask.float()
            negative_txt_distances = text_distances * negatives_mask.float()
            hardest_txt_positive, _ = positive_txt_distances.max(dim=1)
            hardest_txt_negative, _ = negative_txt_distances.min(dim=1)

            positive_img_distances = img_distances * positives_mask.float()
            negative_img_distances = img_distances * negatives_mask.float()
            hardest_img_positive, _ = positive_img_distances.max(dim=1)
            hardest_img_negative, _ = negative_img_distances.min(dim=1)
        
            return (torch.nn.functional.relu(hardest_img_positive + margin - hardest_negative)).mean()

    def total_loss(self, labels, img_embeddings, txt_embeddings, mmr_logits, margin=1.0, instance_weight=1.0, sem_weight=1.0, itm_weight=1.0):
        sem_loss = instance_semantic_loss(img_embeddings, txt_embeddings, labels, margin, mode='semantic')
        inst_loss = instance_semantic_loss(img_embeddings, txt_embeddings, labels, margin, mode='instance')
        itm_loss = itm_loss(mmr_logits, labels)
        return (sem_weight * sem_loss) + (instance_weight * inst_loss) + (itm_weight * itm_loss)

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
        # self.ITEM = nn.ModuleList([TDB(self.hidden_dim, self.num_heads) for _ in range(self.ITEM_lyrs)])
        self.ITEM = nn.ModuleList([TDB(self.hidden_dim, self.num_heads) for _ in range(self.ITEM_lyrs)])
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
        # enhanced_image_tokens = self.ITEM(tgt=enhanced_image_tokens, memory=recipe_focus).transpose(0, 1)
        enhanced_image_tokens = image_tokens  
        for im_layer in self.ITEM:
            enhanced_image_tokens = im_layer(enhanced_image_tokens, recipe_tokens)

        # MTD: The recipe tokens are fed to MTD as Q and the enhanced image tokens as K and V. Then the modalities are fused
        enhanced_recipe_tokens = recipe_tokens.transpose(0, 1)
        for re_layer in self.MTD:
            enhanced_recipe_tokens = re_layer(enhanced_recipe_tokens, enhanced_image_tokens)

        # Project the two embeddings into space for ITM loss
        recipe_projection = self.recipe_proj(enhanced_recipe_tokens.transpose(0, 1)[:, 0, :]) 
        image_projection = self.image_proj(enhanced_image_tokens.transpose(0, 1)[:, 0, :]) 
        concat_projections = torch.cat(recipe_projection, image_projection)
        
        # After fusing the two modalities an ITM loss is applied --> not the triplet loss?
        logits = self.match_score(torch.sigmoid(concat_projections))

        return logits
        

    def seed_torch(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
      


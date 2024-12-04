# CITATIONS: basis for MMR section is https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9857479, A4, 
# https://github.com/mshukor/TFood/blob/main/recipe1m/visu/modality_to_modality_top5.py#L120 ,
# https://github.com/mshukor/TFood/blob/main/recipe1m/models/networks/trijoint.py#L22
# The entirety of this script is based heavily on these

# Imports
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
    def __init__(self, instance_weight=1.0, sem_weight=0.1, itm_weight=1.0):
        super().__init__()
        # self.margin = margin
        # self.margin_step = margin_step
        # self.margin_max = margin_max
        self.instance_weight = instance_weight
        self.sem_weight = sem_weight
        self.itm_weight = itm_weight
        self.itm_loss_lyr = nn.BCELoss()  # CITATION: https://github.com/mshukor/TFood/blob/main/bootstrap.pytorch/bootstrap/templates/default/project/models/criterions/criterion.py

    def fast_distance(self,A,B):
        # SOURCE: COPIED FROM https://github.com/mshukor/TFood/
        # A and B must have norm 1 for this to work for the ranking
        
        A = torch.nn.functional.normalize(A, p=2, dim=-1)
        B = torch.nn.functional.normalize(B, p=2, dim=-1)

        if A.dim() == 2 and B.dim() == 2:
            return torch.mm(A, B.t()) * -1
        elif A.dim() == 3 and B.dim() == 3:
            batch_size_A, na, ms = A.size()
            batch_size_B, nb, mb = B.size()

            # ok we actually need to flatten everything bc otherwise the matrix is the wrong shape :/
            A_flat = A.view(batch_size_A, -1)  # Shape: (batch_size_A, n * m)
            B_flat = B.view(batch_size_B, -1)  # Shape: (batch_size_B, n * m)

            # Compute pairwise distances (batch_size_A x batch_size_B)
            return torch.mm(A_flat, B_flat.t()) * -1


    def itm_loss_from_logits(self, similarity_matrix):
        """
        Compute Image-Text Matching (ITM) Loss using a precomputed similarity matrix.
        Args:
            similarity_matrix (torch.Tensor): Shape (batch_size, batch_size), 
                                            similarity between all image-text pairs.
        Returns:
            torch.Tensor: Loss ITM.
        """
        batch_size = similarity_matrix.size(0)

        #Extract positive scores(diagonal)
        positive_scores = similarity_matrix.diag()  # Positive pairs

        #Find hardest negatives
        hardest_negatives_img = similarity_matrix.masked_fill(
            torch.eye(batch_size, device=similarity_matrix.device).bool(), 
            float('-inf')
        ).max(dim=1).values

        hardest_negatives_txt = similarity_matrix.masked_fill(
            torch.eye(batch_size, device=similarity_matrix.device).bool(), 
            float('-inf')
        ).max(dim=0).values 
        
        #Compute binary cross-entropy loss
        positive_labels = torch.ones_like(positive_scores)
        negative_labels_img = torch.zeros_like(hardest_negatives_img)
        negative_labels_txt = torch.zeros_like(hardest_negatives_txt)
        positive_loss = F.binary_cross_entropy_with_logits(positive_scores, positive_labels)
        negative_loss_img = F.binary_cross_entropy_with_logits(hardest_negatives_img, negative_labels_img)
        negative_loss_txt = F.binary_cross_entropy_with_logits(hardest_negatives_txt, negative_labels_txt)

        # Total ITM loss
        total_loss = positive_loss + negative_loss_img + negative_loss_txt
        return total_loss
    
    def itm_loss(self, mmr_out):
        # Litm = −EtR,tI∼D[y log(s(tR, tI ))+ (1 − y) log(1 − s(tR, tI ))] --> it is just the BCE loss
        labels = torch.eye(mmr_out.size()[0], device=mmr_out.device)
        # print(f'labs {labels.shape}, mmr_sig {torch.sigmoid(mmr_out)}')
        return self.itm_loss_lyr(torch.sigmoid(mmr_out), labels)
        #return self.itm_loss_lyr(mmr_out, labels)
        
    # Measures similarity between an anchor and its directly associated positive while ensuring dissimilarity from negatives in the same batch.
    # "The semantic loss Lsem is the same as the instance loss except for the selection of positive and negative samples"
    # Encourages embeddings to reflect class-level s, tf_labels='base'emantics, ensuring that embeddings from the same class are closer, even if they are not direct pairs.
    def instance_semantic_loss(self, img_embeddings, txt_embeddings, labels, margin=1.0, mode='instance'):
        '''
        mode is instance or semantic,
        image embadggings of shape batch, hidden
        text embeddings of size batch, hidden, 
        labels of size batch, 1
        '''
        # l(xa, xp, xn, α)=[d(xa, xp (positive)) + α − d(xa, xn (negative))]+
        batch_size = img_embeddings.size()[0]
        sim_img_txt = img_embeddings @ txt_embeddings.T
        sim_txt_img = sim_img_txt.T

        if mode == 'instance':
            # Here we need to check that the images are getting the right images and text and vice versa
            distances = self.fast_distance(img_embeddings, txt_embeddings)
            positives = torch.eye(batch_size, device=img_embeddings.device)
            loss_im_txt = F.cross_entropy(-distances, positives.argmax(dim=1))
            loss_txt_im = F.cross_entropy(-distances.T, positives.argmax(dim=1))
            
            return (loss_im_txt + loss_txt_im).mean()
            
        if mode == 'semantic':
            # Here, we focus on having the meaning extracted from the image match the meaning extracted by the text
            # STep 1 is to calculate the similarity between the two embeddings. 
            # Get the same class/value
            labels = torch.eye(batch_size, device=img_embeddings.device)
            # print(f'labels: {labels.shape}')
            # labels = labels.unsqueeze(1)
            positives_mask = labels
            negatives_mask = 1 - positives_mask
    
            txt_embeddings = F.normalize(txt_embeddings, p=2, dim=-1)
            txt_distances  = self.fast_distance(txt_embeddings, txt_embeddings)

            img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)
            img_distances = self.fast_distance(img_embeddings, img_embeddings)

            # Not sure if we need to be comparing to itself but we need to compare to eachother
            img_txt_distances = self.fast_distance(img_embeddings, txt_embeddings)
            txt_img_distances = img_txt_distances.T

            # print(f'img_txt dist: {img_txt_distances.shape}')
            # print(f'img_emb dist: {img_embeddings.shape}')
            # print(f'txt_emb dist: {txt_embeddings.shape}')
            
            # Get positive and negative matches
            positive_img_txt_distances = img_txt_distances * positives_mask
            negative_img_txt_distances = img_txt_distances * negatives_mask 
            hardest_img_txt_positive = positive_img_txt_distances.max(dim=1)[0]
            hardest_img_txt_negative = negative_img_txt_distances.min(dim=1)[0]
            img_txt = torch.nn.functional.relu(hardest_img_txt_positive + margin - hardest_img_txt_negative).mean()

            positive_txt_img_distances = txt_img_distances * positives_mask.float()
            negative_txt_img_distances = txt_img_distances * negatives_mask.float() 
            hardest_txt_img_positive = positive_txt_img_distances.max(dim=1)[0]
            hardest_txt_img_negative = negative_txt_img_distances.min(dim=1)[0]
            txt_img = torch.nn.functional.relu(hardest_txt_img_positive + margin - hardest_txt_img_negative).mean()

            # positive_txt_distances = txt_distances * positives_mask.float()
            # negative_txt_distances = txt_distances * negatives_mask.float()
            # hardest_txt_positive = positive_txt_distances.max(dim=1)[0]
            # hardest_txt_negative = negative_txt_distances.min(dim=1)[0]
            # txt_txt = torch.nn.functional.relu(hardest_txt_positive + self.margin - hardest_txt_negative)).mean())

            # positive_img_distances = img_distances * positives_mask.float()
            # negative_img_distances = img_distances * negatives_mask.float()
            # hardest_img_positive = positive_img_distances.max(dim=1)[0]
            # hardest_img_negative = negative_img_distances.min(dim=1)[0]
            # img_img = torch.nn.functional.relu(hardest_img_positive + self.margin - hardest_img_negative)).mean() 
        
            # Push the positives and negatives apart
            return img_txt + txt_img 

    def total_loss(self, labels, img_embeddings, txt_embeddings, mmr_logits, margin=1.0, tf_labels='base'):
        if tf_labels == 'base':
           tf_labels = torch.ones(img_embeddings.size()[0])
        sem_loss = self.instance_semantic_loss(img_embeddings, txt_embeddings, labels, margin=margin, mode='semantic')
        inst_loss = self.instance_semantic_loss(img_embeddings, txt_embeddings, labels, margin=margin, mode='instance')
        #itm_loss = self.itm_loss(mmr_logits)
        itm_loss = self.itm_loss_from_logits(mmr_logits)
        print(f'itm: {itm_loss} sem: {sem_loss} inst: {inst_loss}')
        return (self.sem_weight * sem_loss) + (self.instance_weight * inst_loss) + (self.itm_weight * itm_loss)

    def total_eval_loss(self, labels, img_embeddings, txt_embeddings):
        sem_loss = self.instance_semantic_loss(img_embeddings, txt_embeddings, labels, mode='semantic')
        inst_loss = self.instance_semantic_loss(img_embeddings, txt_embeddings, labels, mode='instance')
        # Potentially adda classification loss
        return (self.sem_weight * sem_loss) + (self.instance_weight * inst_loss)  



class TDB(nn.Module):
    def __init__(self, hidden_dim, heads, hidden_factor=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Self attention --> ouput is attn_output, attn_output_weights, so we only care about the first
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(hidden_dim)
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(hidden_dim)

        # Feed forward layers --> expand and compress to keep shape --> Sequential to only call one thing... ReLU for nonlinearity but they use sigmoid
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * hidden_factor), nn.ReLU(), nn.Linear(hidden_dim * hidden_factor, hidden_dim))
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, src):
        '''
        Self attention,
        cross attention, 
        feed forward and layer normalization layers with
        residual connections, which are repeated N times!
        '''
        # Input: query, key, valuei, outputs attn_output, attn_output_weights
        tgt = self.self_attn(tgt, tgt, tgt)[0]
        tgt = self.norm_sa(tgt)

        # We don't differentiate between key and value per the paper
        tgt = self.cross_attn(tgt, src, src)[0]
        tgt = self.norm_ca(tgt)

        tgt = self.feed_forward(tgt)
        tgt = self.norm_ff(tgt)

        return tgt


class MMR(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=4, ITEM_lyrs=1, MTD_lyrs=4, projection_dim=512):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim  # MUST be the size of inputs
        self.ITEM_lyrs = ITEM_lyrs
        self.MTD_lyrs = MTD_lyrs
        #self.num_classes = num_classes
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
        # self.match_score = nn.Sequential(nn.Linear(self.projection_dim, self.num_classes), nn.Sigmoid())
        
    def forward(self, recipe_tokens, image_tokens):

        # ITEM: Enhance image tokens by focusing on recipe tokens
        enhanced_image_tokens = image_tokens.transpose(0, 1)
        recipe_focus = recipe_tokens.transpose(0, 1)
        # enhanced_image_tokens = self.ITEM(tgt=enhanced_image_tokens, memory=recipe_focus).transpose(0, 1)
        enhanced_image_tokens = image_tokens  
        for im_layer in self.ITEM:
            enhanced_image_tokens = im_layer(enhanced_image_tokens, recipe_tokens)

        # MTD: The recipe tokens are fed to MTD as Q and the enhanced image tokens as K and V. Then the modalities are fused
        enhanced_recipe_tokens = recipe_tokens
        for re_layer in self.MTD:
            enhanced_recipe_tokens = re_layer(enhanced_recipe_tokens, enhanced_image_tokens)


        # Project the two embeddings into space for ITM loss
        recipe_projection = self.recipe_proj(enhanced_recipe_tokens) #.transpose(0, 1)) #[:, 0, :]) 
        image_projection = self.image_proj(enhanced_image_tokens) # .transpose(0, 1)) #[:, 0, :]) 

        # concat_projections = torch.cat((recipe_projection, image_projection), dim=1)
	# concat_projections = torch.sum(
        similarity_matrix = torch.matmul(recipe_projection, image_projection.T)
        # similarity_score = torch.sum(recipe_projection * image_projection, dim=-1)  # Dot product for similarity
        # print(f'post concat: {concat_projections}')
        
        # After fusing the two modalities an ITM loss is applied --> not the triplet loss?
        # logits = torch.sigmoid(similarity_score).mean(-1)  # self.match_score(torch.sigmoid(concat_projections))

        return similarity_matrix
        

    def seed_torch(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
      


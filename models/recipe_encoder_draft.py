import numpy as np
import torch
from torch import nn

class RecipeEncoder(nn.Module):
    def __init__(self, device, vocab_size, max_len, output_size=1024, hidden_dim=512):
        """
        Initialize the recipe encoder
            Args:
                device (str): Name of device to use
                vocab_size (int): Total words across all text inputs
                max_len (int): Longest line in any text input
                output_size (int): Size of output
                hidden_dim (int): Size of hidden layers
        """
        super().__init__()

        self.device = device
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Shared Transformer Encoders
        self.shared_encoder = TransformerEncoder(vocab_size, device, max_length=max_len, output_size=output_size)

        # Linear layer for final output projection
        self.ll_e = nn.Linear(hidden_dim * 3, output_size, device=self.device)

    def forward(self, input):
        """
        Forward pass of the recipe encoder
            Args:
                input (list(tensors)): List of tensors containing title, ingredients, instructions
            Returns: 
                t_R (tensor)
                e_R (tensor)
        """
        title_array, ingredients_array, instructions_array = [x.to(self.device) for x in input]

        # Title processing
        if len(title_array.shape)==3:
            ttl_1 = torch.squeeze(title_array)
        else:
            ttl_1 = title_array
        
        ttl_2 = self.shared_encoder(ttl_1)

        # Ingredients processing
        ing_1_all = torch.stack([self.shared_encoder(ingredients_array[:, i, :]).mean(dim=1) 
                                 for i in range(ingredients_array.shape[1])], dim=1)
        ing_2 = self.shared_encoder(ing_1_all.mean(dim=1).long())

        # Instructions processing
        ins_1_all = torch.stack([self.shared_encoder(instructions_array[:, i, :]).mean(dim=1) 
                                 for i in range(instructions_array.shape[1])], dim=1)
        ins_2 = self.shared_encoder(ins_1_all.mean(dim=1).long())

        # Decoder-like cross-attention through concatenation
        ttl_3 = self.cross_attention(ttl_2, ing_2, ins_2)
        ing_3 = self.cross_attention(ing_2, ttl_2, ins_2)
        ins_3 = self.cross_attention(ins_2, ttl_2, ing_2)

        # Averaging decoder outputs
        avg_title = ttl_3.mean(dim=1)
        avg_ingr = ing_3.mean(dim=1)
        avg_instr = ins_3.mean(dim=1)

        # Concatenating averaged outputs and projecting to final embedding
        cat_avg = torch.cat((avg_title, avg_ingr, avg_instr), dim=1)
        e_R = self.ll_e(cat_avg)

        return torch.cat((ttl_3, ing_3, ins_3), dim=1), e_R

    def cross_attention(self, query, key1, key2):
        """
        Simple cross-attention mechanism with concatenation.
        """
        cat = torch.cat((key1, key2), dim=1)
        decoder = TransformerDecoder(self.shared_encoder.input_size, self.device, max_length=self.max_len)
        return decoder(query, cat)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, device, output_size=1024, hidden_dim=512, num_enc_heads=4, num_enc_layers=2, max_length=43):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        # Transformer Encoder definition
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=num_enc_heads, batch_first=True, device=device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers).to(device)

        # Embedding layers
        self.srcembeddingL = nn.Embedding(input_size, hidden_dim, device=device)
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim, device=device)

    def forward(self, input):
        input = input.to(self.device)
        src_emb = self.srcembeddingL(input)
        pos_range = torch.arange(0, input.size(1), device=self.device).unsqueeze(0)
        pos_embed = self.srcposembeddingL(pos_range)
        src_pos_emb = src_emb + pos_embed

        return self.encoder(src_pos_emb)


class TransformerDecoder(nn.Module):
    def __init__(self, input_size, device, output_size=1024, hidden_dim=512, num_dec_heads=4, num_dec_layers=2, max_length=43):
        super().__init__()

        self.device = device
        # Transformer Decoder definition
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nhead=num_dec_heads, batch_first=True, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers).to(device)

    def forward(self, query, memory):
        return self.decoder(query, memory)
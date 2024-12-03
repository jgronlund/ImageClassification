# Base code adapted from Deep Learning A4:
# models/Transformer.py, FullTrasformerTranslator
import numpy as np

import torch
from torch import nn
import random

class RecipeEncoder(nn.Module):
    def __init__(self, device, vocab_size, max_len, output_size=1024, hidden_dim=512 ):
        """
        Initialize the recipe encoder
            Args:
                device (str): Name of device to use
                vocab_size (int): Total words across all text inputs
                max_len (int): Longest line in any text input
                output_size (int): Size of output
                hidden_dim (int): Size of hidden layers
            
            Returns: 
                t_R (tensor): output tokens
                e_R (tensor): recipe embedding
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.device = device
        self.max_len = max_len

        # Initialize final layers
        # self.ll_t = nn.Linear(self.hidden_dim*3, self.output_size, device=self.device)
        self.ll_e = nn.Linear(self.hidden_dim*3, output_size, device=self.device)


    def forward(self, input):
        """
        Forward pass of the recipe encoder
            Args:
                input (list(tensors)): List of tensors containing title, ingredients, instructions
            
            Returns: 
                t_R (tensor)
                e_R (tensor)
        """
        # process txt based on txt name (title, ingredients, recipe)
        title_array = input[0].to(self.device) # ["Title"]
        ingredients_array = input[1].to(self.device) # ["Ingredients"]
        instructions_array = input[2].to(self.device) # ["Instructions"]

        #### Encoders ####
        ## Title processing
        # Run through 1 encoder
        title_encoder = TransformerEncoder(self.vocab_size, self.device, 
                                           max_length=self.max_len,
                                           output_size=self.output_size)
        if len(title_array.shape)==3:
            ttl_1 = torch.squeeze(title_array)
        else:
            ttl_1 = title_array
        ttl_2 = title_encoder.forward(ttl_1.long())

        ## Ingredients processing
        # Run each line thru first encoder and avg the output
        ing_1_all = torch.zeros_like(ingredients_array, dtype=torch.float)
        # An encoder for each line
        for i in range(ingredients_array.shape[1]):
            ing = ingredients_array[:,i,:]
            ingr_T_encoder = TransformerEncoder(self.vocab_size, self.device, 
                                                max_length=self.max_len,
                                                output_size=self.output_size)
            ing_1_all[:,i,:] = ingr_T_encoder.forward(ing.long()).mean(dim=2)

        ing_1 = ing_1_all.mean(dim=1).long()

        # Run thru second encoder
        ingr_HT_encoder = TransformerEncoder(self.vocab_size, self.device, 
                                             max_length=self.max_len,
                                             output_size=self.output_size)
        ing_2 = ingr_HT_encoder.forward(ing_1)

        ## Instructions processing
        # Run thru first encoder and avg the output
        ins_1_all = torch.zeros_like(instructions_array, dtype=torch.float)
        # An encoder for each line
        for i in range(instructions_array.shape[1]):
            ins = instructions_array[:,i,:]
            instr_T_encoder = TransformerEncoder(self.vocab_size, self.device,
                                                max_length=self.max_len,
                                                output_size=self.output_size)
            ins_1_all[:,i,:] = instr_T_encoder.forward(ins.long()).mean(dim=2)

        ins_1 = ins_1_all.mean(dim=1).long()

        # Run thru second encoder
        instr_HT_encoder = TransformerEncoder(self.vocab_size, self.device,
                                              max_length=self.max_len,
                                              output_size=self.output_size)
        ins_2 = instr_HT_encoder.forward(ins_1)


        #### Decoders ####
        ## Cross Attention through Concatenation
        # ingr as query and instr/title as key and value
        cat = torch.cat((ins_2, ttl_2), dim=1)
        ingr_decoder = TransformerDecoder(self.vocab_size, self.device, max_length=self.max_len)
        ing_3 = ingr_decoder.forward(ing_2, cat) 

        # instr as query and ingr/title as key and value
        cat = torch.cat((ing_2, ttl_2), dim=1)
        instr_decoder = TransformerDecoder(self.vocab_size, self.device, max_length=self.max_len)
        ins_3 = instr_decoder.forward(ins_2, cat)

        # title as query and ingr/instr as key and value
        cat = torch.cat((ing_2, ins_2), dim=1)
        title_decoder = TransformerDecoder(self.vocab_size, self.device, max_length=self.max_len)
        ttl_3 = title_decoder.forward(ttl_2, cat)

        # Take average of each decoder output
        avg_ingr = torch.mean(ing_3, dim=1).to(self.device)
        avg_instr = torch.mean(ins_3, dim=1).to(self.device)
        avg_title = torch.mean(ttl_3, dim=1).to(self.device)

        # Concat the decoder outputs
        t_R = torch.cat((ttl_3, ing_3, ins_3), dim=1).to(self.device)
        # t_R = self.ll_t(cat)


        # Concat the averages and project out
        cat_avg = torch.cat((avg_title, avg_ingr, avg_instr), dim=1).to(self.device)
        e_R = self.ll_e(cat_avg)


        # output=None
        return t_R, e_R

class TransformerEncoder(nn.Module):

    def __init__(self, input_size, device, output_size=1024, hidden_dim=512,
                 num_enc_heads=4,
                #  dim_feedforward=2048,
                 num_enc_layers=2, 
                #  dropout=0.2,
                 max_length=43, ignore_index=1):
        super(TransformerEncoder, self).__init__()

        self.num_heads = num_enc_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        # self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        # seed_torch(0)

        # Initialize Encoder T
        encoder_layer_T = nn.TransformerEncoderLayer(hidden_dim,
                                                     nhead=num_enc_heads,
                                                     batch_first=True,
                                                     device=self.device
                                                     )

        self.encoder = nn.TransformerEncoder(encoder_layer_T,
                                    num_layers=num_enc_layers,
                                    ).to(self.device)
        

        # Initialize embedding lookup
        self.srcembeddingL = nn.Embedding(input_size, hidden_dim, device=self.device) 
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim, device=self.device)


    def forward(self, input):
        """
         This function computes the full Transformer forward pass used during training.

         :param input: a PyTorch tensor of shape (N,T) these are tokenized input sentences
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        
        # embed input and tgt for processing by transformer
        input = input.to(self.device)
        src_emb = self.srcembeddingL(input)
        pos_range = torch.arange(0,self.max_length, device=self.device).repeat(input.shape[0],1)
        pos_embed = self.srcposembeddingL(pos_range)
        src_pos_emb = (src_emb+ pos_embed).to(self.device)

  
        # invoke transformer to generate output
        outputs = self.encoder(src_pos_emb)


        # pass through final layer to generate outputs

        # outputs = self.ll_o(outputs)

        return outputs

class TransformerDecoder(nn.Module):

    def __init__(self, input_size, device, output_size=1024, hidden_dim=512,
                 num_dec_heads=4,
                #  dim_feedforward=2048,
                 num_dec_layers=2,
                #  dropout=0.2,
                 max_length=43, ignore_index=1):
        super(TransformerDecoder, self).__init__()

        self.num_heads = num_dec_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        # self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        # seed_torch(0)

        # Initialize Decoder HTD
        decoder_layer_HTD = nn.TransformerDecoderLayer(hidden_dim,
                                                     nhead=num_dec_heads,
                                                     batch_first=True,
                                                     device=self.device
                                                     )

        self.decoder = nn.TransformerDecoder(decoder_layer_HTD,
                                    num_layers=num_dec_layers,
                                    ).to(self.device)

        

        # Initialize embedding lookup
        # self.srcembeddingL = nn.Embedding(input_size, hidden_dim, device=self.device) 
        # self.tgtembeddingL = nn.Embedding(output_size, hidden_dim, device=self.device) 
        # self.srcposembeddingL = nn.Embedding(max_length, hidden_dim, device=self.device)
        # self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim, device=self.device)


    def forward(self, input, tgt):
        """
         This function computes the full Transformer forward pass used during training.

         :param input: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """

        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        # tgt = self.add_start_token(tgt)

        # tgt = tgt.to(self.device)
        # tgt_emb = self.tgtembeddingL(tgt)
        # pos_range = torch.arange(0,self.max_length, device=self.device).repeat(tgt.shape[0],1)
        # pos_embed = self.tgtposembeddingL(pos_range)
        # tgt_pos_emb = tgt_emb+ pos_embed

  
        # invoke transformer to generate output
        outputs = self.decoder(input,tgt)

        # pass through final layer to generate outputs
        # outputs = self.ll_o(outputs)

        return outputs



def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
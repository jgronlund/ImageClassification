# CITATIONS: basis for MMR section is https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9857479, and A4
# The output representation (tokens) are then projected into
# 2 spaces; the shared latent space where the instance and semantic
# losses are applied, and the multimodal space where the ITM loss
# is applied on top of the MMR module.
# Goal: exploits the interaction between modalities in a novel regularization scheme, while using only unimodal encoders at test time for efficient retrieval.

# Imports
import numpy as np
import torch
from torch import nn
import random


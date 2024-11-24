from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)



class ImageEncoder(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(ImageEncoder, self).__init__()
        self.parent_model = clip_model
        self.image_embedding_dim = clip_model.config.hidden_size
        self.fc1 = nn.Linear(self.image_embedding_dim, num_classes)

    def forward(self, x):
        pass

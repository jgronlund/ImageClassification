# from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch.functional as F

# Load the CLIP model and processor
# model_name = "openai/clip-vit-base-patch16" ##put this in the ipynb file
# model = CLIPModel.from_pretrained(model_name) ##put this in the ipynb file
# processor = CLIPProcessor.from_pretrained(model_name)




class Image_Encoder(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(Image_Encoder, self).__init__()

        self.parent_model = clip_model
        self.image_embedding_dim = clip_model.config.hidden_size
        self.fc1 = nn.Linear(self.image_embedding_dim, num_classes)

    def forward(self, images, image_labels):
        image_encodings = self.parent_model.get_image_features(pixel_values=images)
        # image_encodings = F.normalize(dim=-1, keepdim=True)
        image_encodings = F.normalize(image_encodings, p=2, dim=-1)
        text_features = self.clip_model.get_text_features(input_ids=image_labels['input_ids'],
                                                          attention_mask=image_labels['attention_mask'])
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        image_logits = self.fc1(image_encodings)


        return image_logits, image_encodings, text_features

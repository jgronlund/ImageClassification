from torch.utils.data import Dataset
import torch
from models.image_encoder import Image2Recipe

class Data_Loading(Dataset):
    """
    Class to combine the Images, Labels, Recipes together to be used in combination when inputted into Model
    """
    def __init__(self, preprocessed_images, text_inputs, recipes, image_labels, instructions):
        self.images = preprocessed_images
        self.text_inputs = text_inputs
        self.recipes = recipes
        self.image_labels = image_labels
        self.instructions = instructions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "text_inputs": {key: val[idx] for key, val in self.text_inputs.items()},
            "recipes": {key: val[idx] for key, val in self.recipes.items()},
            "image_labels": self.image_labels[idx],
            "instructions": {key: val[idx] for key, val in self.instructions.items()},
        }



class Runner(object):
    """
    Class designed to run ViT (train, evaluate, plot)
    """

    def __init__(self, **kwargs):
        """
        Initialize ViT
        """
        self.epochs = kwargs['epochs']
        self.train_loader = kwargs['train_loader']
        self.test_loader = kwargs['test_loader']
        self.val_loader = kwargs['val_loader']
        self.optimizer_name = kwargs['optimizer']
        self.device = kwargs['device']

        self.model = Image2Recipe()

        self.optimizer = None
        optimizer = torch.optim.AdamW([
            {"params": self.model.clip_model.parameters(), "lr": 1e-6},
            {"params": self.model.recipe_encoder.parameters(), "lr": 1e-5}, 
            {"params": self.model.image_fc.parameters(), "lr": 1e-5},
            {"params": self.model.recipe_fc.parameters(), "lr": 1e-5},
        ])


        #TODO: Figure out best way to combine Images, Recipes, Instructions
        self.data_loader = {}
        self.data_loader['train'] = self.train_loader
        self.data_loader['eval'] = self.val_loader
        self.test_loader['test_loader'] = self.test_loader
        
        
        self.to(self.device)

    def train(self):
        """
        Train ViT, image encoder, recipe encoder, MMR
        """

        for epoch in range(self.epochs):

            for phase in ['train', 'eval']:

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                for i, batch_data in enumerate(self.data_loader[phase]):
                    #Looping through batches of training data then eval data each epoch

                    #TODO: Add how the recipe, instructions, and titles will be tokenized
                    images, text_inputs, recipes, image_labels, instructions = (
                        batch_data['images'].to(self.device),
                        #{k: v.to(self.device) for k, v in batch_data['text_inputs'].items()},
                        #batch_data['recipes'].to(self.device),
                        batch_data['image_labels'].to(self.device),
                        #batch_data['instructions'].to(self.device),
                    )

                    self.optimizer.zero_grad()

                    if phase == 'train':
                        output = self.model(images, text_inputs, recipes, image_labels, instructions)
                        ##Combine the Recipe Encoder Losses and Image Encoder Losses based on TFOOD
                        loss = None
                        loss.backward()
                        self.optimizer.step()
                    
                    else: ##Eval mode
                        with torch.no_grad():
                            output = self.model(images, text_inputs, recipes, image_labels, instructions)
                            loss = None ##TODO: Complete how we will calculate the loss with these outputted encodings
                            
                            ##Example solution, but I think the paper does it differently:
                            # contrastive = contrastive_loss(image_features, text_features) + \
                            #                 contrastive_loss(recipe_features, text_features) + \
                            #                 contrastive_loss(recipe_features, image_features)
                            # img_loss = self.criterion(image_logits, image_labels)
                            # rcp_loss = self.criterion(recipe_logits, recipe_labels)
        
                            # # Combine losses
                            # loss = alpha * contrastive + beta * (img_loss + rcp_loss)



                    total_loss += loss.item()

                print(f"{phase}: Epoch {epoch+1}, Loss: {total_loss / len(self.data_loader[phase])}")
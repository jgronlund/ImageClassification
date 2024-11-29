from torch.utils.data import Dataset
import torch
from models.image_2_recipe import Image2Recipe
from models.image_encoder import Image_Encoder
from models.recipe_encoder import RecipeEncoder

class Data_Loading(Dataset):
    """
    Class to combine the Images, Labels, Recipes together to be used in combination when inputted into Model
    """
    def __init__(self, tokenized_ingredients, tokenized_instructions, tokenized_titles, image_tensors, image_labels):
        self.ingredients = torch.tensor(tokenized_ingredients, dtype=torch.long)
        self.instructions = torch.tensor(tokenized_instructions, dtype=torch.long)
        self.titles = torch.tensor(tokenized_titles, dtype=torch.long)
        self.images = image_tensors
        self.labels = image_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "ingredients": self.ingredients[idx],
            "instructions": self.instructions[idx],
            "titles": self.titles[idx],
            "images": self.images[idx],
            "labels": self.labels[idx]
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
        # self.train_loader = kwargs['train_loader']
        # self.test_loader = kwargs['test_loader']
        # self.val_loader = kwargs['val_loader']
        self.optimizer_name = kwargs['optimizer']
        self.device = kwargs['device']

        self.tokenized_ingredients = kwargs['ingredient_tokens']
        self.tokenized_instructions = kwargs['instruction_tokens']
        self.tokenized_title = kwargs['title_tokens']
        self.image_tensor = kwargs['image_tensor']
        self.image_labels = kwargs['image_labels']
        self.clip_model = kwargs['clip_model']
        self.vocab_size = kwargs['vocab_size']
        self.max_len = kwargs['max_len']
        num_classes = len(self.image_labels)

        
        self.image_encoder = Image_Encoder(self.clip_model, num_classes)
        self.recipe_encoder = RecipeEncoder(self.vocab_size, self.max_len)
        self.model = Image2Recipe(self.image_encoder, self.recipe_encoder)


        self.optimizer = None
        optimizer = torch.optim.AdamW([
            {"params": self.model.clip_model.parameters(), "lr": 1e-6},
            {"params": self.model.recipe_encoder.parameters(), "lr": 1e-5}, 
            {"params": self.model.image_fc.parameters(), "lr": 1e-5},
            {"params": self.model.recipe_fc.parameters(), "lr": 1e-5},
        ])


        #TODO: Figure out best way to combine Images, Recipes, Instructions
        self.data_loader = Data_Loading(
            self.tokenized_ingredients, 
            self.tokenized_instructions, 
            self.tokenized_title, 
            self.image_tensor, 
            self.image_labels
        )

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
                    recipes, instructions, titles, images, image_labels = (
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
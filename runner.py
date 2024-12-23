from torch.utils.data import Dataset, DataLoader, random_split
import torch
from models.image_2_recipe import Image2Recipe
from models.image_encoder import Image_Encoder
from models.recipe_encoder import RecipeEncoder
from models.MMR import MMR
from models.MMR import MMR_losses
import matplotlib.pyplot as plt

class Data_Loading(Dataset):
    """
    Class to combine the Images, Labels, Recipes together to be used in combination when inputted into Model
    """
    def __init__(self, tokenized_ingredients, tokenized_instructions, tokenized_titles, image_tensors, tokenized_labels):
        self.ingredients = torch.tensor(tokenized_ingredients, dtype=torch.int16)
        self.instructions = torch.tensor(tokenized_instructions, dtype=torch.int16)
        self.titles = torch.tensor(tokenized_titles, dtype=torch.int16)
        self.images = image_tensors
        self.tokenized_labels = tokenized_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "ingredients": self.ingredients[idx],
            "instructions": self.instructions[idx],
            "titles": self.titles[idx],
            "images": self.images[idx],
            "tokenized_labels": {
                "input_ids": self.tokenized_labels['input_ids'][idx].to(dtype=torch.long),
                "attention_mask": self.tokenized_labels['attention_mask'][idx].to(dtype=torch.uint8)
            }
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
        self.optimizer_name = kwargs['optimizer']
        self.device = kwargs['device']
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['learning_rate']

        self.tokenized_ingredients = kwargs['ingredient_tokens']
        self.tokenized_instructions = kwargs['instruction_tokens']
        self.tokenized_title = kwargs['title_tokens']
        self.image_tensor = kwargs['image_tensors']
        self.image_labels = kwargs['image_labels']
        self.clip_model = kwargs['clip_model']
        self.vocab_size = kwargs['vocab_size']
        self.max_len = kwargs['max_len']
        self.instance_weight = kwargs['instance_weight']
        self.sem_weight = kwargs['sem_weight']
        self.itm_weight = kwargs['itm_weight']
        # Pending variable margin calc 
        self.loss_calcs = MMR_losses(margin=1.0, instance_weight=self.instance_weight, sem_weight=self.sem_weight, itm_weight=self.itm_weight)
        num_classes = len(self.image_labels)

        
        self.image_encoder = Image_Encoder(self.device, self.clip_model, num_classes).to(self.device)
        self.recipe_encoder = RecipeEncoder(self.device, self.vocab_size, self.max_len).to(self.device)
        self.mmr = MMR(hidden_dim=self.image_encoder.clip_model.config.projection_dim).to(self.device)
        # MMR varaibles: num_heads=num_heads, ITEM_lyrs=ITEM_lyrs, MTD_lyrs=MTD_lyrs, projection_dim=projection_dim
        self.model = Image2Recipe(self.image_encoder, self.recipe_encoder, self.mmr).to(self.device)


        ##DO we want to tune each of these learning rates for each model?
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.AdamW([
        #     {"params": self.model.image_encoder.parent_model.parameters(), "lr": 1e-6},
        #     {"params": self.model.recipe_encoder.parameters(), "lr": 1e-5}, 
        #     {"params": self.model.image_encoder.fc1.parameters(), "lr": 1e-5},
        #     {"params": self.model.recipe_encoder.ll_e.parameters(), "lr": 1e-5},
        # ])


        #Combine Images, Recipes, Instructions in training and eval datasets
        self.data_total = Data_Loading(
            self.tokenized_ingredients, 
            self.tokenized_instructions, 
            self.tokenized_title, 
            self.image_tensor, 
            self.image_labels
        )
        training_perc = .9
        train_size = int(training_perc * len(self.data_total))
        eval_size = len(self.data_total) - train_size
        train_dataset, eval_dataset = random_split(self.data_total, [train_size, eval_size])
        self.dataloader = {}
        self.dataloader['train'] = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.dataloader['eval'] = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        #Lists to fill up during training and plotted later for learning curves
        self.train_loss_list = []
        self.eval_loss_list = []
        self.eval_acc_list = []
        self.eval_acc_list = []
        print("finished initializing")


    def train(self):
        """
        Train ViT, image encoder, recipe encoder, MMR
        """

        for epoch in range(self.epochs):

            for phase in ['train', 'eval']:
                total_loss = 0
                total_accuracy = 0
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print("updated")
                for i, batch_data in enumerate(self.dataloader[phase]):
                    #Looping through batches of training data then eval data each epoch
                    #TODO: Add how the recipe, instructions, and titles will be tokenized
                    ingredients, instructions, titles, images, image_labels = (
                        batch_data['ingredients'].to(self.device),
                        batch_data['instructions'].to(self.device),
                        batch_data['titles'].to(self.device),
                        batch_data['images'].to(self.device),
                        batch_data['tokenized_labels']
                    )

                    recipe_enc_src = [titles, ingredients, instructions]
                    self.optimizer.zero_grad()

                    if phase == 'train':
                        output = self.model(images, image_labels, recipe_enc_src)
                        ##Combine the Recipe Encoder Losses and Image Encoder Losses based on TFOOD
                        mmr_logits = output["mmr_logits"]
                        image_logits = output["image_logits"]
                        image_embeddings_proj = output["image_embeddings"]
                        recipe_embeddings_proj = output["recipe_embeddings"]

                        # I am assuming that the image and recipe logits give you a classification score that correspond to the label? 
                        # Do we want to add classification to the loss?
                        # Using baseline all image-recipe pairs match btw.. maybe dont wat to do that?
                        loss = self.loss_calcs.total_loss(image_logits, image_embeddings_proj, recipe_embeddings_proj, mmr_logits)
                        self.train_loss_list.append(loss)
                        loss.backward()
                        self.optimizer.step()
                    
                    else: ##Eval mode
                        with torch.no_grad():
                            output = self.model(images, image_labels, recipe_enc_src)
                            loss = self.loss_calcs.total_eval_loss(image_logits, image_embeddings_proj, recipe_embeddings_proj)
                            self.eval_loss_list.append(loss)



                    print(output)
                    # total_loss += loss.item()

                print(f"{phase}: Epoch {epoch+1}, Loss: {total_loss / len(self.dataloader[phase])}")
                break


    ##Waiting on training code to finish
    def plot_learning_loss_curves(self):
        """
        Plot accuracy and loss curves for training and eval accuracy/loss lists (item/epoch)
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_list, label='Training Loss')
        plt.plot(self.eval_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_acc_list, label='Training Accuracy')
        plt.plot(self.eval_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.show()

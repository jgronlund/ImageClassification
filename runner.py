from torch.utils.data import Dataset


class Image2Recipe(Dataset):
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


        self.optimizer = None


        #TODO: Figure out best way to combine Images, Recipes, Instructions
        self.data_loader = {}
        self.data_loader['train'] = self.train_loader
        self.data_loader['eval'] = self.val_loader
        
        
        
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

                    output = self.model(images, text_inputs, recipes, image_labels, instructions)
                    ##Combine the Recipe Encoder Losses and Image Encoder Losses based on TFOOD
                    loss = None

                    self.optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    total_loss += loss.item()

                print(f"{phase}: Epoch {epoch+1}, Loss: {total_loss / len(self.data_loader[phase])}")
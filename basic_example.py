import torch
from models.image_encoder import Image_Encoder
from models.recipe_encoder import RecipeEncoder
from models.MMR import MMR  # Import your updated MMR module
from runner import Runner  # Assuming your Runner class is saved in runner.py
from transformers import AutoTokenizer
from transformers import CLIPModel
model_name = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Mock data for testing (replace with actual preprocessed data)
batch_size = 8
seq_len = 45
hidden_dim = 512
vocab_size = 10000
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _ = clip.load("ViT-B/32", device=device) 

# Mock tokenized inputs
tokenized_ingredients = torch.randint(0, vocab_size, (100, seq_len)).tolist()
tokenized_instructions = torch.randint(0, vocab_size, (100, seq_len)).tolist()
tokenized_titles = torch.randint(0, vocab_size, (100, seq_len)).tolist()
image_tensors = torch.rand(100, 3, 224, 224)  # Simulate 100 images of shape 3x224x224
tokenized_labels = {
    'input_ids': torch.randint(0, 2, (100,)),  # Binary labels
    'attention_mask': torch.ones(100)
}

from runner import Runner as R
kwargs = {
    'epochs': 10,
    'ingredient_tokens': filtered_df['tokenized_ingredients'].to_list(),
    'instruction_tokens': filtered_df['tokenized_instructions'].to_list(),
    'title_tokens': filtered_df['tokenized_titles'].to_list(),
    'image_tensors': filtered_tensors,
    'image_labels': tokenized_labels,
    'device': device,
    'vocab_size': total_vocab_size,
    'max_len': total_max,
    'clip_model': clip_model,
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'batch_size': 4,
    'max_lengths': {
        'ingredient_tokens': max_length_ing,
        'instruction_tokens': max_length_inst,
        'title_tokens': max_length_title
    }

}
image2recipe = R(**kwargs)
image2recipe.train()

# Plot the Learning Curves
runner.plot_learning_loss_curves()

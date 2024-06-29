import os
import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid, save_image
from transformers import DistilBertTokenizer, DistilBertModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

# Constants
THUMBNAILS_DIR = './thumbnail'
METADATA_FILE = './thumbnail/metadata.csv'
TITLE_MAX_LENGTH = 128
CATEGORY_EMBEDDING_WEIGHT = 2.0
CACHE_DIR = "./.cache"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GetImage:
    def __init__(self, image_id: str, resolution: tuple[int, int], thumbnail_dir: str) -> None:
        self.image_id = image_id
        self.resolution = resolution
        self.thumbnail_dir = thumbnail_dir
        self.extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    def get(self) -> torch.Tensor:
        image_path = self._find_image_path()
        if not image_path:
            print(f"Image ID {self.image_id} was not found.")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image at {image_path}.")
            return None
        
        image = cv2.resize(image, self.resolution)
        tensor = ToTensor()(image).to(device)
        return tensor.to(torch.float32)  # Ensure image is in FP32

    def _find_image_path(self) -> str | None:
        for ext in self.extensions:
            image_path = os.path.join(self.thumbnail_dir, f"images/{self.image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        return None

class ThumbnailDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, thumbnail_dir: str, resolution: tuple[int, int]) -> None:
        self.metadata_df = metadata_df
        self.thumbnail_dir = thumbnail_dir
        self.resolution = resolution[::-1]
        self.title_max_length = TITLE_MAX_LENGTH

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=CACHE_DIR)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=CACHE_DIR).to(device)

        self.mean, self.std = self.calculate_normalization_params()

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.metadata_df.iloc[idx]
        image_id = row['Id']
        category = row['Category']
        title = row['Title']

        image_tensor = self._load_image(image_id)
        if image_tensor is None:
            return None
        
        image_tensor = self._normalize(image_tensor)
        
        text_embeddings = self._get_text_embeddings(title, category)

        combined_text = f"{category} {title}"

        return image_tensor, text_embeddings, combined_text

    def _load_image(self, image_id: str) -> torch.Tensor:
        get_image = GetImage(image_id, self.resolution, self.thumbnail_dir)
        return get_image.get()

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0, 1)  # Ensure values are between 0 and 1
        transform = Normalize(mean=self.mean.to(device), std=self.std.to(device))
        normalized = transform(image)
        return normalized.clamp(-1, 1)

    def _get_text_embeddings(self, title: str, category: str) -> torch.Tensor:
        category_emphasis = ' '.join([category] * int(CATEGORY_EMBEDDING_WEIGHT))
        combined_text = f"{category_emphasis} {title}"
        
        inputs = self.tokenizer(combined_text, return_tensors="pt", max_length=self.title_max_length, padding="max_length", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        
        return outputs.last_hidden_state.squeeze(0)

    def calculate_normalization_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.zeros(3, device=device)
        std = torch.zeros(3, device=device)
        valid_images = 0
        for i in range(len(self.metadata_df)):
            image_id = self.metadata_df.iloc[i]['Id']
            image_tensor = self._load_image(image_id)
            if image_tensor is not None:
                mean += image_tensor.mean(dim=(1, 2))
                std += image_tensor.std(dim=(1, 2))
                valid_images += 1
        
        if valid_images > 0:
            mean /= valid_images
            std /= valid_images
        
        return mean.to(torch.float32), std.to(torch.float32)

def logit_normal_timestep_sampling(shape, device):
    u = torch.randn(shape, device=device, dtype=torch.float32)
    return torch.sigmoid(u)

def reverse_normalize(tensor, mean, std):
    mean = mean.to(device).view(1, 3, 1, 1)
    std = std.to(device).view(1, 3, 1, 1)
    tensor = tensor.to(device)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def save_generated_images(images, mean, std, epoch, batch, save_dir, nrow=2):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        image_tensors = reverse_normalize(images, mean, std)
        
        # Check for NaN or Inf values
        if torch.isnan(image_tensors).any() or torch.isinf(image_tensors).any():
            print(f"NaN or Inf detected in generated images at epoch {epoch}, batch {batch}")
            return
        
        image_grid = make_grid(image_tensors, nrow=nrow, normalize=False)

        # Convert to uint8 manually to avoid warning
        image_grid_np = (image_grid.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_batch_{batch+1}.png")
        cv2.imwrite(save_path, cv2.cvtColor(image_grid_np.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error saving generated images: {e}")

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.05,
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.safety_checker = None

vae = pipe.vae

def train(dataloader, dataset, pipe, num_epochs, learning_rate, device, grad_accumulation_steps=4):
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            images, text_embeddings, prompts = batch
            try:
                images = images.to(device).to(torch.float32)
                text_embeddings = text_embeddings.to(device).to(torch.float32)
                
                # Encode images to latents using VAE
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                timesteps = logit_normal_timestep_sampling(images.shape[0], device)
                
                noise = torch.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                with torch.cuda.amp.autocast():
                    noise_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample
                
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float()) / grad_accumulation_steps
                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    scaler.unscale_(optimizer)  # Unscale gradients for gradient clipping or NaN checks
                    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx+1}, Loss: {loss.item() * grad_accumulation_steps}")
                scheduler.step(loss)
            except Exception as e:
                print(f"Error in training loop: {e}")
                continue
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            generated_images = vae.decode(latents).sample  # Extracting the sample attribute here
            save_generated_images(generated_images, dataset.mean, dataset.std, epoch, batch_idx, "./thumbnail/generated")

if __name__ == "__main__":
    metadata_df = pd.read_csv(METADATA_FILE)
    dataset = ThumbnailDataset(metadata_df, THUMBNAILS_DIR, (432, 768))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Increase batch size to 8
    
    train(dataloader, dataset, pipe, num_epochs=10, learning_rate=1e-4, device=device)  # Increase learning rate to 1e-4
    
    title = "Exciting AI Developments"
    category = "Tech"
    text_embeddings = dataset._get_text_embeddings(title, category).unsqueeze(0)
    
    with torch.no_grad():
        latents = pipe.vae.encode(text_embeddings).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents).sample  # Extracting the sample attribute here
    
    save_image(image, "generated_thumbnail.png")
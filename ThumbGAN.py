import os
import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomAffine
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
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
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PREFIX = "checkpoint_epoch"
FINAL_MODEL_PATH = "./final_model"
GENERATED_IMAGE_DIR = "./thumbnail/generated"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
        tensor = ToTensor()(image).to(DEVICE)
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
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=CACHE_DIR).to(DEVICE)

        self.mean, self.std = self.calculate_normalization_params()

        # Define data augmentation transforms
        self.transforms = torch.nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomAffine(degrees=5, translate=(0.1, 0.1))  # Slight rotations and translations
        )

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
        image_tensor = self.transforms(image_tensor)  # Apply data augmentation
        
        text_embeddings = self._get_text_embeddings(title, category)

        combined_text = f"{category} {title}"

        return image_tensor, text_embeddings, combined_text

    def _load_image(self, image_id: str) -> torch.Tensor:
        get_image = GetImage(image_id, self.resolution, self.thumbnail_dir)
        return get_image.get()

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0, 1)  # Ensure values are between 0 and 1
        transform = Normalize(mean=self.mean.to(DEVICE), std=self.std.to(DEVICE))
        normalized = transform(image)
        return normalized.clamp(-1, 1)

    def _get_text_embeddings(self, title: str, category: str) -> torch.Tensor:
        category_emphasis = ' '.join([category] * int(CATEGORY_EMBEDDING_WEIGHT))
        combined_text = f"{category_emphasis} {title}"
        
        inputs = self.tokenizer(combined_text, return_tensors="pt", max_length=self.title_max_length, padding="max_length", truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        
        return outputs.last_hidden_state.squeeze(0)

    def calculate_normalization_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.zeros(3, device=DEVICE)
        std = torch.zeros(3, device=DEVICE)
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
    mean = mean.to(DEVICE).view(1, 3, 1, 1)
    std = std.to(DEVICE).view(1, 3, 1, 1)
    tensor = tensor.to(DEVICE)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def save_generated_images(images, mean, std, epoch, batch, save_dir, nrow=2):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        image_tensors = reverse_normalize(images, mean, std)
        
        # Convert to uint8 and apply contrast enhancement
        image_np = (image_tensors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image_np = image_np.transpose(0, 2, 3, 1)  # NCHW to NHWC
        
        enhanced_images = []
        for img in image_np:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            enhanced_images.append(enhanced)
        
        enhanced_images = np.stack(enhanced_images)
        image_grid = make_grid(torch.from_numpy(enhanced_images.transpose(0, 3, 1, 2)), nrow=nrow, normalize=False)
        
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_batch_{batch+1}.png")
        cv2.imwrite(save_path, cv2.cvtColor(image_grid.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error saving generated images: {e}")

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

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

def train(dataloader, dataset, pipe, num_epochs, learning_rate, device, grad_accumulation_steps=4, checkpoint=None):
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if specified
    if checkpoint:
        try:
            checkpoint_data = torch.load(checkpoint, map_location=device)
            pipe.unet.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            print(f"Loaded checkpoint from {checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting training from scratch.")
    else:
        print(f"No checkpoint provided. Starting training from scratch.")
    
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
        
        # Save checkpoint after each epoch
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{epoch+1}.pt")
        torch.save({
            'model_state_dict': pipe.unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Verify the checkpoint after saving
        try:
            torch.load(checkpoint_path, map_location=DEVICE)
            print(f"Checkpoint verified successfully: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to verify checkpoint: {e}")

        # Generate and save images after each epoch
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            generated_images = vae.decode(latents).sample  # Extracting the sample attribute here
            save_generated_images(generated_images, dataset.mean, dataset.std, epoch, batch_idx, GENERATED_IMAGE_DIR)

    # Save final model weights
    if not os.path.exists(FINAL_MODEL_PATH):
        os.makedirs(FINAL_MODEL_PATH)
    torch.save(pipe.unet.state_dict(), os.path.join(FINAL_MODEL_PATH, "unet_final.pt"))
    print(f"Saved final model weights to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    metadata_df = pd.read_csv(METADATA_FILE)
    dataset = ThumbnailDataset(metadata_df, THUMBNAILS_DIR, (720, 1280))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Specify checkpoint path if available, or use None
    checkpoint_path = "./checkpoints/checkpoint_epoch_13.pt"  # e.g., "./checkpoints/checkpoint_epoch_1.pt" if resuming

    # Verify the checkpoint
    try:
        torch.load(checkpoint_path, map_location=DEVICE)
    except Exception as e:
        print(f"Checkpoint verification failed: {e}")
        checkpoint_path = None  # Ignore checkpoint if invalid

    train(dataloader, dataset, pipe, num_epochs=50, learning_rate=1e-4, device=DEVICE, checkpoint=checkpoint_path)
    
    # Generate new images
    title = "Exciting AI Developments"
    category = "Tech"
    prompt = f"{category} {title}"
    
    num_images_to_generate = 4
    generator = torch.Generator(device=DEVICE).manual_seed(42)  # for reproducibility
    
    with torch.no_grad():
        images = pipe(
            prompt,
            num_images_per_prompt=num_images_to_generate,
            generator=generator,
            guidance_scale=7.5
        ).images
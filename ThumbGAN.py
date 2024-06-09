import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader
import pandas
import Networks
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Constants and hyperparameters
THUMBNAILS_DIR = 'thumbnail'
METADATA_FILE = './thumbnail/metadata.csv'
BATCH_SIZE = 64
START_RESOLUTION = (72, 128)
TARGET_RESOLUTION = (432, 768)
TITLE_MAX_LENGTH = 15
NUMBER_EPOCHS = 3000
SAMPLE_INTERVAL = 100 # Interval to save generated images
GENERATOR_LEARNING_RATE = 0.001
DISCRIMINATOR_LEARNING_RATE = 0.0001
CHECKPOINT_PHASE = 0  # 0 for no checkpoint, 1 for 1/3, 2 for 2/3, 3 for 3/3

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fetch image from the thumbnail directory !!on review!!
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
        tensor = ToTensor()(image)
        return tensor

    def _find_image_path(self) -> str | None:
        for ext in self.extensions:
            image_path = os.path.join(self.thumbnail_dir, f"images/{self.image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        return None
       
#Dataset preparation class !!on review!!
class ThumbnailDataset(Dataset):
    def __init__(self, metadata_df: pandas.DataFrame, thumbnail_dir: str, resolution: tuple[int, int], glove: GloVe) -> None:
        self.metadata_df = metadata_df
        self.thumbnail_dir = thumbnail_dir
        self.resolution = resolution
        self.title_max_length = TITLE_MAX_LENGTH
        self.glove = glove

        # Calculate normalization parameters
        self.mean, self.std = self.calculate_normalization_params()

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata_df.iloc[idx]
        image_id = row['Id']
        category = row['Category']
        title = row['Title']

        # Load and preprocess image
        image_tensor = self._load_image(image_id)
        image_tensor = self._normalize(image_tensor)

        # Get title embedding
        title_indices = self._tokenize_and_embed_title(title)
        title_indices = torch.tensor(title_indices, dtype=torch.long)

        # Get category tensor
        category_tensor = torch.tensor([category], dtype=torch.long)

        return image_tensor, category_tensor, title_indices
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        get_image = GetImage(image_id, self.resolution, self.thumbnail_dir)
        return get_image.get()

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        transform = Normalize(mean=self.mean, std=self.std)
        return transform(image)

    def _tokenize_and_embed_title(self, title: str) -> list[int]:
        tokens = title.split()[:self.title_max_length]
        indices = [self.glove.stoi[token] if token in self.glove.stoi else self.glove.stoi['unk'] for token in tokens]
        indices += [0] * (self.title_max_length - len(indices))
        return indices

    def calculate_normalization_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.zeros(3)
        std = torch.zeros(3)
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
        
        return mean, std

# Weight initialization for generator and discriminator
def weights_init_normal(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Dynamic label smoothing for reducing discriminator overconfidence
def dynamic_label_smoothing(epoch: int, n_epochs: int, initial_smoothing: float=0.1, final_smoothing: float=0.9) -> float:
    return initial_smoothing + (final_smoothing - initial_smoothing) * (epoch / n_epochs)

#Saves and loads epoch, model, optimizer, loss, and resolution checkpoint
def save_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer , loss: float, resolution: tuple[int,int], filename: str) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'resolution': resolution,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, float, tuple[int,int]]:
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    resolution = checkpoint['resolution']
    return epoch, loss, resolution

def reverse_normalize(tensor, mean, std):
    mean = mean.clone().detach().to(device)
    std = std.clone().detach().to(device)

    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor

def save_generated_images(images, mean, std, epoch, batch, save_dir, nrow=5):
    images = reverse_normalize(images, mean, std)
    image_grid = make_grid(images, nrow=nrow, normalize=False)

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the image grid
    save_path = os.path.join(save_dir, f"{epoch+1}_{batch+1}.png")
    save_image(image_grid, save_path)



#File preparation
metadata_df = pandas.read_csv(METADATA_FILE)

#Representing the categories as unique integers
label_encoder = LabelEncoder()
metadata_df['Category'] = label_encoder.fit_transform(metadata_df['Category'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Loading GloVe embeddings for title embedding 
glove = GloVe(name='6B', dim=100)
vocab_size = len(glove.itos)
embedding_dim = glove.dim

# Pretrained ResNet model for generator feature extraction
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
for param in resnet.parameters():
    param.requires_grad = False

# Instantiate the dataset and dataloader
thumbnail_dataset = ThumbnailDataset(metadata_df, THUMBNAILS_DIR, START_RESOLUTION, glove)
dataloader = DataLoader(thumbnail_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(thumbnail_dataset[1])


# Instantiate the embedding network
num_categories = len(metadata_df['Category'].unique())
category_embedding_dim = 50
title_embedding_dim = 100

category_title_embedding_net = Networks.CategoryTitleEmbeddingNet(
    num_categories=num_categories,
    category_embedding_dim=category_embedding_dim,
    vocab_size=vocab_size,
    title_embedding_dim=title_embedding_dim,
    title_max_length=TITLE_MAX_LENGTH
).to(device)


# Instantiate the GAN models
noise_dim = 100 #not sure about this purpose 
generator = Networks.Generator(embedding_dim=1, noise_dim=noise_dim, img_channels=3, img_size=START_RESOLUTION).to(device)
discriminator = Networks.Discriminator(embedding_dim=1, img_channels=3, img_size=START_RESOLUTION).to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE, betas=(0.5, 0.999))

# Training loop
G_losses = []
D_losses = []

new_resolution = START_RESOLUTION

# Load checkpoint if specified
if CHECKPOINT_PHASE > 0:
    #open checkpoint file
    checkpoint_generator = f"checkpoint_generator_epoch_{(CHECKPOINT_PHASE * NUMBER_EPOCHS // 3)}.pth"
    checkpoint_discriminator = f"checkpoint_discriminator_epoch_{(CHECKPOINT_PHASE * NUMBER_EPOCHS // 3)}.pth"
    
    #Load checkpoint
    start_epoch, _, new_resolution = load_checkpoint(checkpoint_generator, generator, optimizer_G)
    _, _, _ = load_checkpoint(checkpoint_discriminator, discriminator, optimizer_D)
    thumbnail_dataset.resolution = new_resolution
    generator.update_img_size(new_resolution)
    discriminator.update_img_size(new_resolution)
else:
    start_epoch = 0


# Training loop
for epoch in range(start_epoch, NUMBER_EPOCHS):
    smoothing_factor = dynamic_label_smoothing(epoch, NUMBER_EPOCHS)
    for i, (imgs, category_tensor, title_indices) in enumerate(dataloader):
        # Prepare batch
        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)
        category_tensor = category_tensor.to(device)
        title_indices = title_indices.to(device)

        # Prepare labels
        valid = smoothing_factor * torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Reset gradients
        optimizer_D.zero_grad()

        # Generate embeddings for real images
        embeddings_real = category_title_embedding_net(category_tensor, title_indices).squeeze().to(device)

        # Forward pass real images
        real_validity = discriminator(embeddings_real, real_imgs)
        real_loss = adversarial_loss(real_validity, valid)

        # Generate fake images for discriminator 
        noise = torch.randn(batch_size, noise_dim).to(device)
        embeddings_fake = category_title_embedding_net(category_tensor, title_indices).squeeze().to(device)
        gen_imgs = generator(embeddings_fake, noise)

        # Forward pass fake images
        fake_validity = discriminator(embeddings_fake, gen_imgs.detach())
        fake_loss = adversarial_loss(fake_validity, fake)

        # Compute total discriminator loss and train discriminator
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Reset Generator gradients
        optimizer_G.zero_grad()

        # Generate fake images again for generator training
        noise = torch.randn(batch_size, noise_dim).to(device)
        embeddings_fake = category_title_embedding_net(category_tensor, title_indices).squeeze().to(device)
        gen_imgs = generator(embeddings_fake, noise)
        fake_validity = discriminator(embeddings_fake, gen_imgs)

        # Compute generator loss and train generator
        g_loss = adversarial_loss(fake_validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # Save losses for evaluation    
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        # Periodically save generated images
        if i % SAMPLE_INTERVAL == 0:
            save_generated_images(gen_imgs.data[:25], thumbnail_dataset.mean, thumbnail_dataset.std, epoch, i, f"{THUMBNAILS_DIR}/generated", nrow=5)

    print(f"[Epoch {epoch+1}/{NUMBER_EPOCHS}] [Batch {i+1}/{len(dataloader)}] [D loss: {d_loss.item():.3f}] [G loss: {g_loss.item():.3f}]")

    # Progressive growing step for resolution (also save checkpoints)
    if epoch != 0 and epoch % (NUMBER_EPOCHS // 3) == 0 and new_resolution != TARGET_RESOLUTION:
        # Save checkpoints
        save_checkpoint(epoch, generator, optimizer_G, g_loss.item(), new_resolution, f"./checkpoints/checkpoint_generator_epoch_{epoch}.pth")
        save_checkpoint(epoch, discriminator, optimizer_D, d_loss.item(), new_resolution, f"./checkpoints/checkpoint_discriminator_epoch_{epoch}.pth")

        # Update resolution
        new_resolution = (new_resolution[0] * 2, new_resolution[1] * 2)
        thumbnail_dataset.resolution = new_resolution
        generator.update_img_size(new_resolution)
        discriminator.update_img_size(new_resolution)

        if (new_resolution == TARGET_RESOLUTION):     
            dataloader = DataLoader(thumbnail_dataset, batch_size=2, shuffle=True)
        else:
            dataloader = DataLoader(thumbnail_dataset, batch_size=dataloader.batch_size // 8, shuffle=True) 
# Plot losses
plt.figure()
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
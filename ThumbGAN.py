import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image as torchvision_save_image
from torchvision.transforms import ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader
import pandas
import Networks
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torchvision.utils import save_image

# Constants
THUMBNAILS_DIR = 'thumbnail'
METADATA_FILE = './thumbnail/metadata.csv'
BATCH_SIZE = 64
START_RESOLUTION = (128, 72)
TARGET_RESOLUTION = (256, 144)
TITLE_MAX_LENGTH = 15

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Preparation
class GetImage:
    def __init__(self, image_id:str, resolution:tuple[int, int], thumbnail_dir:str) -> None:
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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resolution)
        tensor = ToTensor()(image)
        return tensor

    def _find_image_path(self) -> str | None:
        for ext in self.extensions:
            image_path = os.path.join(self.thumbnail_dir, f"images/{self.image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        return None
    
class ThumbnailDataset(Dataset):
    def __init__(self, metadata_df:pandas.DataFrame , thumbnail_dir:str, resolution:tuple[int,int], glove:GloVe) -> None:
        self.metadata_df = metadata_df
        self.thumbnail_dir = thumbnail_dir
        self.resolution = resolution
        self.title_max_length = TITLE_MAX_LENGTH
        self.glove = glove

        # Calculate normalization parameters
        self.mean, self.std = self.calculate_normalization_params()

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata_df.iloc[idx]
        image_id = row['Id']
        category = row['Category']
        title = row['Title']

        # Load and preprocess image
        image_tensor = self._load_image(image_id) #CAN BE NONE
        image_tensor = self._normalize(image_tensor)

        # Get title embedding
        title_indices = self._tokenize_and_embed_title(title)
        title_indices = torch.tensor(title_indices, dtype=torch.long)

        # Get category tensor
        category_tensor = torch.tensor([category], dtype=torch.long)

        return image_tensor, category_tensor, title_indices
    
    def _load_image(self, image_id:str) -> torch.Tensor:
        get_image = GetImage(image_id, self.resolution, self.thumbnail_dir)
        return get_image.get()

    def _normalize(self, image:torch.Tensor) -> torch.Tensor:
        transform = Normalize(mean=self.mean, std=self.std)
        return transform(image)

    def _tokenize_and_embed_title(self, title:str) -> list[int]:
        tokens = title.split()[:self.title_max_length]
        indices = [self.glove.stoi[token] if token in self.glove.stoi else self.glove.stoi['unk'] for token in tokens]
        indices += [0] * (self.title_max_length - len(indices))
        return indices

    def calculate_normalization_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate mean and std of pixel values across the dataset
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i in range(len(self.metadata_df)):
            image_id = self.metadata_df.iloc[i]['Id']
            image_tensor = self._load_image(image_id)
            if (image_tensor is not None):
                mean += image_tensor.mean(dim=(1, 2))
                std += image_tensor.std(dim=(1, 2))
            else:
                break

            mean /= len(self.metadata_df)
            std /= len(self.metadata_df)
            return mean, std
        
metadata_df = pandas.read_csv(METADATA_FILE)
label_encoder = LabelEncoder()
metadata_df['Category'] = label_encoder.fit_transform(metadata_df['Category'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


# Load GloVe embeddings
glove = GloVe(name='6B', dim=100)
vocab_size = len(glove.itos)
embedding_dim = glove.dim

# Pretrained ResNet model for feature extraction
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
for param in resnet.parameters():
    param.requires_grad = False

# Instantiate the dataset and dataloader
thumbnail_dataset = ThumbnailDataset(metadata_df, THUMBNAILS_DIR, START_RESOLUTION, glove)
dataloader = DataLoader(thumbnail_dataset, batch_size=BATCH_SIZE, shuffle=True)

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
noise_dim = 100
generator = Networks.Generator(embedding_dim=category_embedding_dim + title_embedding_dim * TITLE_MAX_LENGTH, noise_dim=noise_dim, img_channels=3, img_size=START_RESOLUTION[0]).to(device)
discriminator = Networks.Discriminator(embedding_dim=category_embedding_dim + title_embedding_dim * TITLE_MAX_LENGTH, img_channels=3, img_size=START_RESOLUTION[0]).to(device)

# Loss and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
n_epochs = 5
sample_interval = 100

def progressive_resize(image:torch.Tensor, target_resolution: tuple[int,int]) -> torch.Tensor:
    return Resize(target_resolution)(image)

for epoch in range(n_epochs):
    for i, (imgs, category_tensor, title_indices) in enumerate(dataloader):
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
        fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)
        real_imgs = imgs.to(device)
        category_tensor = category_tensor.to(device)
        title_indices = title_indices.to(device)

        # Generate embeddings
        embeddings = category_title_embedding_net(category_tensor, title_indices).squeeze().to(device)

        # Train Generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, noise_dim).to(device)
        gen_imgs = generator(embeddings, noise)
        g_loss = adversarial_loss(discriminator(embeddings, gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(embeddings, real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(embeddings, gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Print progress
        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated samples
        if epoch % sample_interval == 0:
            torchvision_save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5, normalize=True)

    # Progressive growing step: Increase resolution every few epochs
    if epoch % (n_epochs // 4) == 0 and epoch != 0:
        new_resolution = (START_RESOLUTION[0] * 2, START_RESOLUTION[1] * 2)
        thumbnail_dataset.resolution = new_resolution
        for param in resnet.parameters():
            param.requires_grad = True
        generator.img_size = new_resolution[0]
        discriminator.img_size = new_resolution[0]


# Streamlit for visualization
st.title("Generated Thumbnails")
st.write("Select a category and enter a title to generate a thumbnail:")

category = st.selectbox("Category", options=range(num_categories))
title = st.text_input("Title")
generate_button = st.button("Generate Thumbnail")

if generate_button:
    title_indices = torch.tensor(thumbnail_dataset._tokenize_and_embed_title(title), dtype=torch.long).unsqueeze(0).to(device)
    category_tensor = torch.tensor([category], dtype=torch.long).to(device)
    embeddings = category_title_embedding_net(category_tensor, title_indices).squeeze().to(device)
    noise = torch.randn(1, noise_dim).to(device)
    with torch.no_grad():
        gen_img = generator(embeddings, noise).cpu()
    st.image(gen_img.squeeze().permute(1, 2, 0).numpy(), clamp=True)
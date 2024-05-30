import os
import cv2
import torch
import torchtext
from torchtext.vocab import GloVe
from torchvision import models
from torch.utils.data import Dataset
import pandas
import Networks

#Do NOT change unless you know what you're doing
THUMBNAILS_DIR = 'thumbnail' 
METADATA_FILE = './thumbnail/metadata.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GetImage:
    def __init__(self, image_id, image_width, image_height, thumbnail_dir):
        self.image_id = image_id
        self.image_width = image_width
        self.image_height = image_height
        self.thumbnail_dir = thumbnail_dir
        self.extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    def get(self):
        image_path = self._find_image_path()
        if not image_path:
            print(f"Image ID {self.image_id} was not found.")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image at {image_path}.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_width, self.image_height))
        tensor = torch.tensor(image, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1)
        tensor = self.normalize(tensor)
        return tensor

    def _find_image_path(self):
        for ext in self.extensions:
            image_path = os.path.join(self.thumbnail_dir, f"images\{self.image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        return None

    def normalize(self, image):
        image /= 255.0
        mean = torch.mean(image, dim=[1, 2], keepdim=True)
        std = torch.std(image, dim=[1, 2], keepdim=True)
        normalized_image = (image - mean) / std
        return normalized_image
    
class ThumbnailDataset(Dataset):
    def __init__(self, metadata_df, thumbnail_dir, image_width, image_height, vocab_size, title_max_length, glove, resnet):
        self.metadata_df = metadata_df
        self.thumbnail_dir = thumbnail_dir
        self.image_width = image_width
        self.image_height = image_height
        self.title_max_length = title_max_length
        self.glove = glove
        self.resnet = resnet.eval()  # Ensure the resnet model is in eval mode

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_id = row['ID']
        category = row['Category']
        title = row['Title']

        # Load and preprocess image, create title indices, and make category tensor
        image_tensor = self._load_image(image_id)
        title_indices = tokenize_and_embed_title(title, self.title_max_length)
        title_indices = torch.tensor(title_indices, dtype=torch.long)
        category_tensor = torch.tensor([category], dtype=torch.long)

        # Extract image features using pretrained ResNet and combine into embedding
        image_features = self._extract_image_features(image_tensor)
        embedding_vector = torch.cat((image_features, category_tensor.float(), title_indices.float()), dim=0)

        return image_tensor, embedding_vector

    def _load_image(self, image_id):
        get_image = GetImage(image_id, self.image_width, self.image_height, self.thumbnail_dir)
        return get_image.get()

    def _extract_image_features(self, image_tensor):
        with torch.no_grad():
            features = self.resnet(image_tensor.unsqueeze(0)).squeeze()
        return features
    
def tokenize_and_embed_title(title, max_length=15):
    tokens = title.split()[:max_length]
    indices = [glove.stoi[token] if token in glove.stoi else glove.stoi['unk'] for token in tokens]
    indices += [0] * (max_length - len(indices))
    return indices

metadata_df = pandas.read_csv(METADATA_FILE)

# Load GloVe embeddings
glove = GloVe(name='6B', dim=100)
vocab_size = len(glove.itos)
embedding_dim = glove.dim

# Pretrained ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
for param in resnet.parameters():
    param.requires_grad = False


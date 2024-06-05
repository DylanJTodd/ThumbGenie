import numpy
import torch
from torch import nn
from torch.nn import functional as Fa

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


                        ###############################################
                        ### Convert a batch of title and categories ###
                        ###       into dense, embedded vectors      ###
                        ###############################################

class CategoryTitleEmbeddingNet(nn.Module):
    def __init__(self, num_categories:int, category_embedding_dim:int, vocab_size:int, title_embedding_dim:int, title_max_length:int):
        """
        Initializes the title / category embedding neural network

        Args:
            num_categories:          Total number of unique categories.
            category_embedding_dim:  Dimensions of the category embedding vector.
            vocab_size:              Total number of unique tokens.
            title_embedding_dim:     Dimensions of the title embedding vector.
            title_max_length:        Maximum number of tokens allowed in the title.

        Returns:
            None
        """
        super(CategoryTitleEmbeddingNet, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)
        self.title_embedding = nn.Embedding(vocab_size, title_embedding_dim)

        self.fully_connected1 = nn.Linear(category_embedding_dim + title_embedding_dim * title_max_length, 256)
        self.fully_connected2 = nn.Linear(256, 128)
        self.fully_connected3 = nn.Linear(128, 64)
        self.fully_connected4 = nn.Linear(64, 1)
        
    def forward(self, category_indices:torch.Tensor, title_indices:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network. Does the work of converting the category(ies) and title(s) to a single dense embedded vector

        Args:
            category_indices:       A tensor of unique numerical indices that represent the category
            title_indices:          A tensor of unique numerical indices that represent vocab words in the title.

        Returns:
            final_output:           A dense vector representing the title and category
        """

        category_embedded = self.category_embedding(category_indices)
        title_embedded = self.title_embedding(title_indices)
        title_embedded = title_embedded.view(title_embedded.size(0), -1)
        category_embedded = category_embedded.view(category_embedded.size(0), -1)
    

        combined_embeddings = torch.cat((category_embedded, title_embedded), dim=1)
        combined_output = torch.relu(self.fully_connected1(combined_embeddings))
        combined_output = torch.relu(self.fully_connected2(combined_output))
        combined_output = torch.relu(self.fully_connected3(combined_output))
        final_output = self.fully_connected4(combined_output)
        return final_output
    
class Generator(nn.Module):
    def __init__(self, embedding_dim, noise_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.fc1 = nn.Linear(embedding_dim + noise_dim, 4*4*512)
        self.bn1 = nn.BatchNorm1d(4*4*512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc5 = nn.Linear(64, img_channels * img_size[0] * img_size[1])
        self.tanh = nn.Tanh()

    def update_img_size(self, img_size):
        self.img_size = img_size
        self.fc5 = nn.Linear(64, self.img_channels * self.img_size[0] * self.img_size[1]).to(self.fc5.weight.device)

    def forward(self, embedding, noise):
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(1)
        elif embedding.dim() == 0:
            embedding = embedding.unsqueeze(0).unsqueeze(1)

        x = torch.cat((embedding, noise), dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 512, 4, 4)  # Reshape to match the expected input size of the convolutional layer

        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))

        x = self.pool(x)  # Apply adaptive average pooling to reduce the spatial dimensions to (1, 1)
        x = x.view(x.size(0), -1)  # Flatten the output to (batch_size, 64)
        x = self.fc5(x)  # Fully connected layer to get the final image
        x = self.tanh(x)
        x = x.view(-1, self.img_channels, self.img_size[0], self.img_size[1])
        return x
    
class Generator(nn.Module):
    def __init__(self, embedding_dim, noise_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.fc1 = nn.Linear(embedding_dim + noise_dim, 4*4*512)
        self.bn1 = nn.BatchNorm1d(4*4*512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0, device='cuda')
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, device='cuda')
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, device='cuda')
        self.bn4 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc5 = nn.Linear(64 * 8 * 8, img_channels * img_size[0] * img_size[1])
        self.tanh = nn.Tanh()

    def update_img_size(self, img_size):
        self.img_size = img_size
        self.fc5 = nn.Linear(64 * 8 * 8, self.img_channels * self.img_size[0] * self.img_size[1]).to(self.fc5.weight.device)

    def forward(self, embedding, noise):
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(1)
        elif embedding.dim() == 0:
            embedding = embedding.unsqueeze(0).unsqueeze(1)

        x = torch.cat((embedding, noise), dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 512, 4, 4)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.tanh(x)
        x = x.view(-1, self.img_channels, self.img_size[0], self.img_size[1])
        return x

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim + img_channels * img_size[0] * img_size[1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def update_img_size(self, img_size):
        self.img_size = img_size
        self.fc1 = nn.Linear(self.embedding_dim + self.img_channels * self.img_size[0] * self.img_size[1], 512).to(self.fc1.weight.device)

    def forward(self, embedding, img):
        img_flat = img.view(img.size(0), -1)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(1)
        elif embedding.dim() == 0:
            embedding = embedding.unsqueeze(0).unsqueeze(1)
        x = torch.cat((embedding, img_flat), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

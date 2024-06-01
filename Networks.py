import numpy
import torch
from torch import nn
from torch.nn import functional


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
        self.img_size = img_size
        self.img_channels = img_channels
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim + noise_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, img_channels * img_size[0] * img_size[1])
        self.tanh = nn.Tanh()

    def update_img_size(self, img_size):
        self.img_size = img_size
        self.fc4 = nn.Linear(512, self.img_channels * self.img_size[0] * self.img_size[1]).to(self.fc4.weight.device)

    def forward(self, embedding, noise):
        embedding.unsqueeze_(1)
        x = torch.cat((embedding, noise), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = x.view(-1, 3, self.img_size[0], self.img_size[1])
        return x
    

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim + img_channels * img_size[0] * img_size[1], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def update_img_size(self, img_size):
        self.img_size = img_size
        self.fc1 = nn.Linear(self.embedding_dim + self.img_channels * self.img_size[0] * self.img_size[1], 512).to(self.fc1.weight.device)

    def forward(self, embedding, img):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((embedding, img_flat), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

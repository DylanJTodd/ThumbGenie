import os
import cv2
import torch
import requests

THUMBNAILS_DIR = 'thumbnail'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GetImage:
    def __init__(self, image_id, image_width, image_height):
        self.image_id = image_id
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = self.find_image_path()

    def find_image_path(self):
        for root, dirs, files in os.walk(f"{THUMBNAILS_DIR}/images", topdown=False):
            for file in files:
                if file[:-4] == self.image_id:
                    return os.path.join(root, file)
        return None

    def get(self):
        if not self.image_path:
            print(f"Image ID {self.image_id} was not found.")
            return None

        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Failed to read image at {self.image_path}.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
        image = cv2.resize(image, (self.image_width, self.image_height))
        tensor = torch.tensor(image, dtype=torch.float32)
        tensor = tensor.permute(2, 1, 0)  # Change to (channels, width, height)
        tensor = self.normalize(tensor)
        return tensor

    def normalize(self, image):
        image /= 255.0  # Normalize to range [0, 1]
        mean = torch.mean(image, dim=[1, 2], keepdim=True)
        std = torch.std(image, dim=[1, 2], keepdim=True)
        normalized_image = (image - mean) / std
        return normalized_image


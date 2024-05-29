import os
import cv2
import torch
import requests

THUMBNAILS_DIR = 'thumbnail'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_image_path_dict(THUMBNAILS_DIR):
    image_path_dict = {}
    for root, dirs, files in os.walk(f"{THUMBNAILS_DIR}/images", topdown=False):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Ensure you're capturing valid image files
                image_id = file[:-4]  # Assuming the ID is the filename without the extension
                image_path_dict[image_id] = os.path.join(root, file)
    return image_path_dict

class GetImage:
    def __init__(self, image_id, image_width, image_height, image_path_dict):
        self.image_id = image_id
        self.image_width = image_width
        self.image_height = image_height
        self.image_path_dict = image_path_dict

    def get(self):
        image_path = self.image_path_dict.get(self.image_id)
        if not image_path:
            print(f"Image ID {self.image_id} was not found.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image at {image_path}.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
        image = cv2.resize(image, (self.image_width, self.image_height))
        tensor = torch.tensor(image, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1)  # Change to (channels, width, height)
        tensor = self.normalize(tensor)
        return tensor

    def normalize(self, image):
        image /= 255.0  # Normalize to range [0, 1]
        mean = torch.mean(image, dim=[1, 2], keepdim=True)
        std = torch.std(image, dim=[1, 2], keepdim=True)
        normalized_image = (image - mean) / std
        return normalized_image

image_path_dict = create_image_path_dict(THUMBNAILS_DIR)
image_getter = GetImage(image_id='MnmrEMbDdyA', image_width=128, image_height=72, image_path_dict=image_path_dict)
image_tensor = image_getter.get()
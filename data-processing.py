import os
import cv2
import torch

THUMBNAILS_DIR = 'thumbnail'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class getImage:
    def __init__(self, image_id):
        self.image_id = image_id

    def get(self):
        tensor = None
        for root, dirs, files in os.walk(f"{THUMBNAILS_DIR}/images", topdown=False):
            for file in files:
                if file[:-4] == self.image_id:
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        tensor = torch.tensor(image, dtype=torch.float32)
                        tensor = tensor.permute(2, 0, 1)  # Change to (channels, width, height)
                        tensor = self.normalize(tensor)
                    break
        if tensor is None:
            print("ID was not found.")
            return None
        return tensor

    def normalize(self, image):
        image /= 255.0  # Normalize to range [0, 1]
        mean = torch.mean(image, dim=[1, 2], keepdim=True)
        std = torch.std(image, dim=[1, 2], keepdim=True)
        normalized_image = (image - mean) / std
        return normalized_image



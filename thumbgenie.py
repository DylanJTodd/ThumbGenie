import os
import torch
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import get_peft_model, LoraConfig
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import save_image

# Constants
TITLE_MAX_LENGTH = 128
CATEGORY_EMBEDDING_WEIGHT = 2.0

MEDIAFIRE_URL = "https://www.mediafire.com/file/hxj2h3gn0y6ibtz/unet_final-001.pt/file" #Default trained model
CACHE_DIR = "./.cache"
GENERATED_IMAGE_DIR = "./generated_images"
CATEGORIES = ["science", "news", "food", "blog", "tech", "informative", "comedy", "entertainment", "automobile", "videogames"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurable constants
IMAGE_RESOLUTION = (720, 1280)  # Height, Width USE SAME RESOLUTION AS TRAINING IMAGES
FINAL_MODEL_PATH = None  # None as default to use the basic trained model. BE AWARE: WAS NOT TRAINED WELL DUE TO LIMITED RESOURCES

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGE_DIR, exist_ok=True)

# Function to get the direct download link from Mediafire if using default model
def get_mediafire_direct_link(mediafire_url: str) -> str:
    response = requests.get(mediafire_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    download_link = soup.find('a', {'id': 'downloadButton'})
    if download_link:
        return download_link['href']
    else:
        raise Exception("Could not find the download link on the page")

# Function to download the file
def download_file(url: str, save_path:str ) -> None:
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Check if FINAL_MODEL_PATH is None and update it
if FINAL_MODEL_PATH is None:
    print("Final Model Path set as 'None'... Downloading default model...")
    
    FINAL_MODEL_PATH = "./final_model/unet_final-001.pt"
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    direct_link = get_mediafire_direct_link(MEDIAFIRE_URL)
    download_file(direct_link, FINAL_MODEL_PATH)

    print(f"Default model has been downloaded and saved to {FINAL_MODEL_PATH}")
else:
    if not os.path.exists(FINAL_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {FINAL_MODEL_PATH}")

#Use DistilBert pretrained tokenizer and model to preprocess text
def preprocess_text(title: str, category: str) -> torch.Tensor:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=CACHE_DIR)
    text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=CACHE_DIR).to(DEVICE)
    
    category_emphasis = ' '.join([category] * int(CATEGORY_EMBEDDING_WEIGHT))
    combined_text = f"{category_emphasis} {title}"
    
    inputs = tokenizer(combined_text, return_tensors="pt", max_length=TITLE_MAX_LENGTH, padding="max_length", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    
    return outputs.last_hidden_state.squeeze(0)

#Loads the pretrained image generation model
def load_model() -> StableDiffusionPipeline:
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
    pipe.unet.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
    pipe.unet.eval()
    pipe.safety_checker = None
    
    return pipe

#Generates images based on the given prompt
def generate_images(pipe, prompt, num_images=4, generator_seed=42) -> np.ndarray:
    generator = torch.Generator(device=DEVICE).manual_seed(generator_seed)
    
    with torch.no_grad():
        images = pipe(
            prompt,
            num_images_per_prompt=num_images,
            generator=generator,
            guidance_scale=7.5,
            height=IMAGE_RESOLUTION[0],
            width=IMAGE_RESOLUTION[1],
        ).images
    return images

def save_generated_images(images: np.ndarray, save_dir: str) -> None:
    for i, image in enumerate(images):
        image_tensor = ToTensor()(image).unsqueeze(0)
        image_resized = Resize((IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]))(image_tensor).squeeze(0)

        image_path = os.path.join(save_dir, f"generated_image_{i+1}.png")
        save_image(image_resized, image_path)
        print(f"Saved generated image: {image_path}")

#Get user input for title, categories and number of images to generate
def get_user_input():
    while True:
        print("Enter '0' to quit at any time.")
        title = input(f"Enter title (max {TITLE_MAX_LENGTH} characters): ").strip()
        if title == "0":
            return None, None
        
        if len(title) > TITLE_MAX_LENGTH:
            print(f"Title is too long. Please enter a title with at most {TITLE_MAX_LENGTH} characters.")
            continue

        print("Categories: " + ", ".join(CATEGORIES))
        category1 = input("Select first category: ").strip().lower()
        if category1 == "0":
            return None, None

        category2 = input("Select second category (or press Enter to use the same category): ").strip().lower()
        if category2 == "0":
            return None, None
        
        if category1 not in CATEGORIES or (category2 and category2 not in CATEGORIES):
            print("Invalid category selection. Please choose from the provided list.")
            continue

        if not category2:
            category2 = category1

        categories = f"{category1} {category2}" if category1 != category2 else category1

        batch_size = input("Enter number of images to be generated (default is 1): ").strip()
        if batch_size == "0":
            batch_size = 1
        return title, categories, batch_size

if __name__ == "__main__":
    while True:
        title, categories, batch_size = get_user_input()
        if title is None and categories is None:
            print("Exiting...")
            break
        
        prompt = f"{categories} {title}"
        pipe = load_model()
        images = generate_images(pipe, prompt, num_images=batch_size)
        save_generated_images(images, GENERATED_IMAGE_DIR)
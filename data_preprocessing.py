import os
import shutil
import requests
import zipfile
from PIL import Image

# Define the main directory paths
thumbnails_dir = 'thumbnail'
images_dir = os.path.join(thumbnails_dir, 'images')

# Create the images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

# Download the dataset
url = "https://tinyurl.com/ThumbnailDataset"
print("Downloading dataset...")
response = requests.get(url)
with open("dataset.zip", "wb") as file:
    file.write(response.content)

print("Extracting...")
# Extract the dataset
with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall(thumbnails_dir)
    
# Delete the dataset.zip file after extraction
os.remove("dataset.zip")

# List of common image file extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Path to the thumbnails images directory
thumbnails_images_dir = os.path.join(thumbnails_dir, 'thumbnails', 'images')

# Function to generate a unique file path if a file with the same name already exists
def get_unique_path(dest_path):
    base, extension = os.path.splitext(dest_path)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = f"{base}_{counter}{extension}"
        counter += 1
    return dest_path

# Function to resize image to the given dimensions
def resize_image(image_path, size=(1280, 720)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)
        img.save(image_path)

# Move all images to the main images directory and resize them
print("Organizing...")
for root, dirs, files in os.walk(thumbnails_images_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(images_dir, file)
            dest_path = get_unique_path(dest_path)  # Ensure unique file path
            try:
                shutil.move(src_path, dest_path)
                resize_image(dest_path)  # Resize image to 1280 x 720
            except FileNotFoundError as e:
                print(f"Error moving file {src_path} to {dest_path}: {e}")

# Move the metadata.csv file to the root of thumbnails_dir
metadata_src = os.path.join(thumbnails_dir, 'thumbnails', 'metadata.csv')
metadata_dest = os.path.join(thumbnails_dir, 'metadata.csv')
if os.path.exists(metadata_src):
    shutil.move(metadata_src, metadata_dest)

print("Deleting...")
# Clean up: remove any empty directories in the original thumbnails images directory
for root, dirs, files in os.walk(thumbnails_images_dir, topdown=False):
    for name in dirs:
        dir_path = os.path.join(root, name)
        if not os.listdir(dir_path):
            os.rmdir(dir_path)

# Remove the empty 'thumbnails/images' directory if it exists
if not os.listdir(thumbnails_images_dir):
    os.rmdir(thumbnails_images_dir)

# Remove the empty 'thumbnails' directory if it exists
thumbnails_subdir = os.path.join(thumbnails_dir, 'thumbnails')
if not os.listdir(thumbnails_subdir):
    os.rmdir(thumbnails_subdir)
print("Concatenating...")

#for 

print("All operations completed successfully!")
print("")
print(f"Images are in {thumbnails_dir}/images, and metadata can be found in {thumbnails_dir}/metadata.csv ")

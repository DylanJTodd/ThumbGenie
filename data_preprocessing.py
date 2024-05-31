import os
import pandas
import shutil
import requests
import zipfile
from PIL import Image

THUMBNAILS_DIR = 'thumbnail' #Don't change this
images_dir = os.path.join(THUMBNAILS_DIR, 'images')

os.makedirs(images_dir, exist_ok=True)

# Download the dataset
url = "https://tinyurl.com/YoutubeThumbnailLink"
print("Downloading dataset...")
response = requests.get(url)
with open("dataset.zip", "wb") as file:
    file.write(response.content)

print("Extracting...")
# Extract the dataset
with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall(THUMBNAILS_DIR)
    
# Delete the dataset.zip file after extraction
os.remove("dataset.zip")

image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

thumbnails_images_dir = os.path.join(THUMBNAILS_DIR, 'thumbnails', 'images')

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
            try:
                shutil.move(src_path, dest_path)
                resize_image(dest_path)  # Resize image to 1280 x 720
            except FileNotFoundError as e:
                print(f"Error moving file {src_path} to {dest_path}: {e}")

# Move the metadata.csv file to the root of THUMBNAILS_DIR
metadata_src = os.path.join(THUMBNAILS_DIR, 'thumbnails', 'metadata.csv')
metadata_dest = os.path.join(THUMBNAILS_DIR, 'metadata.csv')
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
thumbnails_subdir = os.path.join(THUMBNAILS_DIR, 'thumbnails')
if not os.listdir(thumbnails_subdir):
    os.rmdir(thumbnails_subdir)

print("Cleaning up metadata...")

# Load the metadata file
metadata_path = os.path.join(THUMBNAILS_DIR, 'metadata.csv')
metadata = pandas.read_csv(metadata_path)

# Check for each row if the corresponding image file exists
def image_exists(row):
    for ext in image_extensions:
        image_path = os.path.join(images_dir, f"{row['Id']}{ext}")
        if os.path.exists(image_path):
            return True
    return False

# Filter the metadata to keep only rows with existing images
cleaned_metadata = metadata[metadata.apply(image_exists, axis=1)]

# Save the cleaned metadata back to the file
cleaned_metadata.to_csv(metadata_path, index=False)

print("All operations completed successfully!")
print("")
print(f"Images are in {THUMBNAILS_DIR}/images, and metadata can be found in {THUMBNAILS_DIR}/metadata.csv ")

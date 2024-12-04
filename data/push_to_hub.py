import json
import glob
import os
from datasets import Dataset, Image, Value, Sequence
from PIL import Image as PILImage
import numpy as np
from huggingface_hub import HfApi

def load_annotations(annotations_dir, images_dir):
    data = {
        'image': [],
        'image_hash': [],
        'point_id': [],
        'name': [],
        'coordinates': [],
        'resolution': []
    }
    
    # Get all JSON files in the annotations directory
    json_files = glob.glob(os.path.join(annotations_dir, '*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            annotation = json.load(f)
            
        # Get corresponding image path
        image_path = os.path.join(images_dir, annotation['image_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue
            
        # Process each element and its names
        for element in annotation['elements']:
            for name in element['names']:
                data['image'].append(image_path)
                data['image_hash'].append(annotation['image_hash'])
                data['point_id'].append(element['id'])
                data['name'].append(name)
                data['coordinates'].append(element['coordinates'])
                data['resolution'].append(annotation['image_size'])
    
    return data

def main():
    # Define paths
    annotations_dir = '../tmp/data/annotations'
    images_dir = '../tmp/data/screenshots'
    
    # Load and process annotations
    print("Loading annotations...")
    data = load_annotations(annotations_dir, images_dir)
    
    # Create dataset
    print("Creating dataset...")
    dataset = Dataset.from_dict(data)
    
    # Cast coordinates and resolution to the correct type
    dataset = dataset.cast_column('coordinates', Sequence(feature=Value('float32'), length=2))
    dataset = dataset.cast_column('resolution', Sequence(feature=Value('int32'), length=2))
    
    # Convert image paths to actual images
    dataset = dataset.cast_column('image', Image())
    
    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    dataset.push_to_hub(
        "agentsea/anchor",
        private=True,
    )
    
    print("Done!")

if __name__ == "__main__":
    main()

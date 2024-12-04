import json
import glob
import os
from datasets import Dataset, Image, Value, Sequence
from PIL import Image as PILImage
import numpy as np
from huggingface_hub import HfApi
import random

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
            
        # Load image using PIL
        try:
            image = PILImage.open(image_path)
            image = image.convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue
            
        # Process each element and its names
        for element in annotation['elements']:
            for name in element['names']:
                data['image'].append(image)  # Store PIL Image object directly
                data['image_hash'].append(annotation['image_hash'])
                data['point_id'].append(element['id'])
                data['name'].append(name)
                data['coordinates'].append(element['coordinates'])
                data['resolution'].append(annotation['image_size'])
    
    return data

def create_splits(data):
    """Create train/test splits by randomly selecting one name per element for test set"""
    # Group data by image and point_id
    grouped_data = {}
    for i in range(len(data['image'])):
        key = (data['image_hash'][i], data['point_id'][i])  # Use image_hash instead of PIL image object
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(i)
    
    # Calculate target number of test examples (10% of total)
    total_examples = len(data['image'])
    target_test_size = total_examples // 10
    
    # Get unique images and shuffle them
    all_images = list(set(data['image_hash']))  # Use image_hash instead of PIL image object
    random.shuffle(all_images)
    
    # Initialize train and test indices
    train_indices = []
    test_indices = []
    
    # Keep adding images to test set until we reach target size
    test_images = set()
    current_test_size = 0
    
    for img in all_images:
        # Count how many new examples this image would add to test set
        potential_test_examples = sum(
            1 for (image, _) in grouped_data.keys() 
            if image == img
        )
        
        # Add to test set if it won't exceed target by too much
        if current_test_size < target_test_size:
            test_images.add(img)
            current_test_size += potential_test_examples
    
    # Now process all groups
    for (image, _), indices in grouped_data.items():
        if image in test_images:
            # For test images, randomly select one name for test set
            test_idx = random.choice(indices)
            test_indices.append(test_idx)
            # Add remaining indices to train set
            train_indices.extend([idx for idx in indices if idx != test_idx])
        else:
            # Add all indices to train set for non-test images
            train_indices.extend(indices)
    
    print(f"Split sizes - Train: {len(train_indices)}, Test: {len(test_indices)} "
          f"(Total: {len(train_indices) + len(test_indices)})")
    
    # Create split datasets
    train_data = {k: [v[i] for i in train_indices] for k, v in data.items()}
    test_data = {k: [v[i] for i in test_indices] for k, v in data.items()}
    
    return train_data, test_data

def main():
    # Define paths
    annotations_dir = '../tmp/data/annotations'
    images_dir = '../tmp/data/screenshots'
    
    # Load and process annotations
    print("Loading annotations...")
    data = load_annotations(annotations_dir, images_dir)
    
    # Create train/test splits
    print("Creating train/test splits...")
    train_data, test_data = create_splits(data)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    # Cast columns for both datasets
    for dataset in [train_dataset, test_dataset]:
        dataset = dataset.cast_column('coordinates', Sequence(feature=Value('float32'), length=2))
        dataset = dataset.cast_column('resolution', Sequence(feature=Value('int32'), length=2))
        dataset = dataset.cast_column('image', Image())
    
    # Create DatasetDict
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    dataset_dict.push_to_hub(
        "agentsea/anchor",
        private=True,
    )
    
    print("Done!")

if __name__ == "__main__":
    main()

import json
import glob
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import os
from tqdm import tqdm
from PIL import Image

def load_and_process_files(directory):
    all_data = []
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    # Iterate through each JSON file with a progress bar
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Load the image
            try:
                image = Image.open(data['image_path'])
                # Get image resolution
                width, height = image.size
            except Exception as e:
                print(f"Error loading image {data['image_path']}: {e}")
                continue
            
            # Process each bounding box and element name combination
            for bb in data['bounding_boxes']:
                # Calculate center point
                center_x = (bb['x1'] + bb['x2']) / 2
                center_y = (bb['y1'] + bb['y2']) / 2
                
                # Create bounding box array
                bbox = [bb['x1'], bb['y1'], bb['x2'], bb['y2']]
                
                # Create an entry for each element name
                for element_name in bb['element_names']:
                    processed_entry = {
                        'image_id': data['image_id'],
                        'image': image,
                        'resolution': [width, height],
                        'bb_id': bb['bb_id'],
                        'bbox': bbox,
                        'point': [center_x, center_y],
                        'element_name': element_name
                    }
                    all_data.append(processed_entry)
    
    return all_data

def main():
    # Load and process all JSON files
    data = load_and_process_files('data/bounding_boxes')
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Create dataset dictionary with a single split
    dataset_dict = DatasetDict({
        'train': dataset  # You might want to split this into train/val/test later
    })
    
    # Push to hub
    dataset_dict.push_to_hub(
        "agentsea/anchor",
        private=False,
        token=os.environ.get('HF_TOKEN')  # Make sure to set this environment variable
    )

if __name__ == "__main__":
    main()

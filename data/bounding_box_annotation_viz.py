import gradio as gr
import json
import os
import cv2
import numpy as np
from pathlib import Path

class AnnotationVisualizer:
    def __init__(self):
        self.current_index = 0
        self.images = []
        self.annotations = {}
        self.bb_data = {}
        
        # Load images and annotations
        self.load_data()
        
    def load_data(self):
        # Load images
        self.images = list(Path("data/images").glob("*.[pj][np][g]"))
        if not self.images:
            raise Exception("No images found in data/images/")
            
        # Load bounding box data
        for bb_file in Path("data/bounding_boxes").glob("*.json"):
            with open(bb_file) as f:
                data = json.load(f)
                self.bb_data[data["image_id"]] = data
                
        # Load annotations from JSONL file
        annotation_file = Path("data/batch_files/output_1.jsonl")
        if annotation_file.exists():
            with open(annotation_file) as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            data = json.loads(line)  # Use loads instead of load
                            bb_id = data["custom_id"]
                            content = data["response"]["body"]["choices"][0]["message"]["content"]
                            self.annotations[bb_id] = content
                        except json.JSONDecodeError:
                            print(f"Error parsing line: {line}")
                            continue

    def draw_annotations(self, image_path):
        """Draw bounding boxes and labels on image"""
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image ID from path
        image_id = Path(image_path).stem
        
        # Get bounding boxes for this image
        if image_id not in self.bb_data:
            return image
            
        boxes = self.bb_data[image_id]["bounding_boxes"]
        
        # Draw each box and its label
        for box in boxes:
            x1, y1 = int(box["x1"]), int(box["y1"])
            x2, y2 = int(box["x2"]), int(box["y2"])
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Get and draw label
            if box["bb_id"] in self.annotations:
                label = self.annotations[box["bb_id"]]
                # Add background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1-20), (x1+w, y1), (255, 0, 0), -1)
                # Add text
                cv2.putText(image, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return image

    def navigate(self, direction):
        """Navigate through images"""
        if direction == "next":
            self.current_index = (self.current_index + 1) % len(self.images)
        else:  # prev
            self.current_index = (self.current_index - 1) % len(self.images)
            
        image_path = self.images[self.current_index]
        return self.draw_annotations(image_path)

def create_ui():
    visualizer = AnnotationVisualizer()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Annotation Visualization")
        
        with gr.Row():
            with gr.Column():
                image_output = gr.Image(
                    label="Annotated Image",
                    type="numpy"
                )
                
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
        
        # Set up navigation handlers
        prev_btn.click(
            fn=lambda: visualizer.navigate("prev"),
            outputs=image_output
        )
        
        next_btn.click(
            fn=lambda: visualizer.navigate("next"),
            outputs=image_output
        )
        
        # Show first image on startup
        if visualizer.images:
            image_output.value = visualizer.draw_annotations(visualizer.images[0])
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()

import gradio as gr
import os
import json
from datetime import datetime
import uuid
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image as PILImage
from uuid import uuid4
import base64
from openai import OpenAI
from typing import Dict

class RealtimeBoundingBoxAnnotator:
    def __init__(self):
        self.current_index = 0
        self.bbox_data = []
        self.current_image_path = None
        self.current_image_id = None
        self.is_drawing = False
        self.current_box = None
        self.image_to_boxes = {}
        self.image_to_id = {}
        self.client = OpenAI()  # Initialize OpenAI client
        
        # Create necessary directories
        os.makedirs("data/images", exist_ok=True)
        os.makedirs("data/bounding_boxes", exist_ok=True)
        os.makedirs("data/input_images", exist_ok=True)
        os.makedirs("data/crops", exist_ok=True)
        
        # Load available images
        self.load_available_images()
        
        # Update default prompts to request multiple names
        self.system_prompt = "You are an expert at identifying UI elements in images."
        self.user_prompt = "Provide 2-3 alternative names for the UI element shown in the image. The names should be completely non-ambiguous. That is, if there are two buttons that look identical, they should have different names. For example, if there are two sign in buttons then differentiate them with additional information. Return the names as a comma-separated list."

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def crop_image(self, image_path: str, bbox: Dict) -> PILImage.Image:
        """Crop image according to bounding box coordinates"""
        with PILImage.open(image_path) as img:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            return img.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))

    def get_element_name(self, image_path: str, bbox: Dict) -> list:
        """Get element names from OpenAI API"""
        # Create temporary directory for crops if it doesn't exist
        temp_dir = Path("data/crops")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Crop and save temporary image
        cropped = self.crop_image(image_path, bbox)
        temp_path = temp_dir / f"{bbox['bb_id']}.jpg"
        cropped.save(temp_path, "JPEG")
        
        # Encode both full and cropped images
        full_image_base64 = self.encode_image(image_path)
        cropped_image_base64 = self.encode_image(temp_path)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{full_image_base64}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{cropped_image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": self.user_prompt
                            }
                        ]
                    }
                ],
                max_tokens=100  # Increased for multiple names
            )
            
            # Parse comma-separated response into list
            element_names = [
                name.strip() 
                for name in response.choices[0].message.content.strip().split(',')
            ]
        except Exception as e:
            element_names = [f"Error: {str(e)}"]
        finally:
            temp_path.unlink()
            
        return element_names

    def handle_select(self, image, evt: gr.SelectData):
        """Handle mouse events for drawing boxes with real-time annotation"""
        if not self.is_drawing:
            # Start drawing
            self.current_box = {
                'bb_id': str(uuid4()),
                'x1': evt.index[0],
                'y1': evt.index[1]
            }
            self.is_drawing = True
        else:
            # Finish drawing
            if self.current_box:
                self.current_box.update({
                    'x2': evt.index[0],
                    'y2': evt.index[1]
                })
                
                # Get element names from API
                element_names = self.get_element_name(self.current_image_path, self.current_box)
                
                # Add element names to box data
                self.current_box['element_names'] = element_names
                
                self.bbox_data.append(self.current_box)
                self.current_box = None
            self.is_drawing = False
            
        updated_image = self.draw_boxes_on_image(image)
        return updated_image, self.format_bbox_text()

    def format_bbox_text(self):
        """Format bounding box data for display"""
        return "\n".join([
            f"Box {i+1}: ({b['x1']}, {b['y1']}) to ({b['x2']}, {b['y2']}) - {', '.join(b.get('element_names', ['Processing...']))}"
            for i, b in enumerate(self.bbox_data)
        ])

    def parse_bbox_text(self, text: str):
        """Parse the textbox content back into bbox data, preserving coordinates but updating names"""
        if not text.strip():
            return
            
        current_boxes = {i: box for i, box in enumerate(self.bbox_data)}
        
        for line in text.split('\n'):
            if not line.strip():
                continue
                
            try:
                # Parse line like "Box 1: (100, 200) to (300, 400) - Name1, Name2, Name3"
                box_num = int(line.split(':')[0].replace('Box ', '')) - 1
                names_part = line.split(' - ', 1)[1].strip()
                names_list = [name.strip() for name in names_part.split(',')]
                
                if box_num in current_boxes:
                    current_boxes[box_num]['element_names'] = names_list
            except (ValueError, IndexError):
                print(f"Warning: Could not parse line: {line}")
                continue
        
        self.bbox_data = [current_boxes[i] for i in sorted(current_boxes.keys())]

    def draw_boxes_on_image(self, image_path):
        """Draw all bounding boxes on the image with labels"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return image_path

        for box in self.bbox_data:
            cv2.rectangle(
                image,
                (int(box['x1']), int(box['y1'])),
                (int(box['x2']), int(box['y2'])),
                (255, 0, 0),
                2
            )
            
            # Add first element name as label (to avoid cluttering)
            if 'element_names' in box and box['element_names']:
                label = box['element_names'][0][:20] + '...' if len(box['element_names'][0]) > 20 else box['element_names'][0]
                cv2.putText(
                    image,
                    label,
                    (int(box['x1']), int(box['y1'] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )
        
        # Draw current box if being drawn
        if self.is_drawing and self.current_box:
            cv2.rectangle(
                image,
                (int(self.current_box['x1']), int(self.current_box['y1'])),
                (int(self.current_box['x1']), int(self.current_box['y1'])),
                (0, 255, 0),  # Green color
                2
            )

        return image

    # Include all other methods from the original BoundingBoxAnnotator class...
    def load_available_images(self):
        """Load list of images from input directory"""
        self.image_files = [
            f for f in os.listdir("data/input_images")
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_files.sort()

    def load_existing_annotations(self, image_path):
        """Load existing annotations for an image"""
        image_id = self.find_existing_image_id(image_path)
        if image_id:
            self.current_image_id = image_id
            self.image_to_id[image_path] = image_id
            annotation_path = f"data/bounding_boxes/{image_id}.json"
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                    return data['bounding_boxes']
        return []

    def find_existing_image_id(self, image_path):
        """Find existing image ID by comparing files"""
        if not image_path:
            return None
            
        image_filename = os.path.basename(image_path)
        for filename in os.listdir("data/images"):
            if filename.endswith(os.path.splitext(image_filename)[1]):
                existing_path = os.path.join("data/images", filename)
                if self.compare_images(image_path, existing_path):
                    return os.path.splitext(filename)[0]
        return None

    def compare_images(self, path1, path2):
        """Compare two images to check if they're identical"""
        if not (os.path.exists(path1) and os.path.exists(path2)):
            return False
        with PILImage.open(path1) as img1, PILImage.open(path2) as img2:
            return list(img1.getdata()) == list(img2.getdata())

    def navigate_images(self, direction, bbox_text: str):
        """Navigate through available images"""
        if not self.image_files:
            return None, "No images available"
            
        self.save_annotations(bbox_text, auto_save=True)
            
        if direction == 'next':
            self.current_index = (self.current_index + 1) % len(self.image_files)
        else:  # prev
            self.current_index = (self.current_index - 1) % len(self.image_files)
            
        image_path = os.path.join("data/input_images", self.image_files[self.current_index])
        self.current_image_path = image_path
        
        self.current_image_id = self.image_to_id.get(image_path, None)
        
        if image_path in self.image_to_boxes:
            self.bbox_data = self.image_to_boxes[image_path]
        else:
            self.bbox_data = self.load_existing_annotations(image_path)
            self.image_to_boxes[image_path] = self.bbox_data
        
        updated_image = self.draw_boxes_on_image(image_path)
        return updated_image, self.format_bbox_text()

    def save_annotations(self, bbox_text: str, auto_save=False):
        """Save bounding box data and image"""
        if not self.current_image_path:
            return "No image loaded" if not auto_save else None
            
        # Update annotations from textbox
        self.parse_bbox_text(bbox_text)
            
        if not self.bbox_data:
            return "No data to save" if not auto_save else None
            
        self.image_to_boxes[self.current_image_path] = self.bbox_data
            
        if self.current_image_path in self.image_to_id:
            self.current_image_id = self.image_to_id[self.current_image_path]
        elif not self.current_image_id:
            self.current_image_id = str(uuid4())
            self.image_to_id[self.current_image_path] = self.current_image_id
            
            _, ext = os.path.splitext(self.current_image_path)
            new_image_path = f"data/images/{self.current_image_id}{ext}"
            shutil.copy2(self.current_image_path, new_image_path)
        
        filename = f"data/bounding_boxes/{self.current_image_id}.json"
        data = {
            "image_id": self.current_image_id,
            "image_path": f"data/images/{self.current_image_id}{os.path.splitext(self.current_image_path)[1]}",
            "bounding_boxes": self.bbox_data,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
            
        return f"Saved {len(self.bbox_data)} bounding boxes to {filename}" if not auto_save else None

    def clear_boxes(self):
        """Clear boxes for current image only"""
        self.bbox_data = []
        self.image_to_boxes[self.current_image_path] = []
        return self.draw_boxes_on_image(self.current_image_path), ""

    def update_prompts(self, system_prompt, user_prompt):
        """Update the system and user prompts"""
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return f"Prompts updated successfully. Next box will use new prompts."

def create_ui():
    annotator = RealtimeBoundingBoxAnnotator()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Real-time Bounding Box Annotator")
        gr.Markdown("Click and drag on the image to draw bounding boxes. Each box will be automatically annotated.")
        
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    label="Image",
                    type="filepath",
                    interactive=True,
                    value=None
                )
                with gr.Row():
                    prev_btn = gr.Button("Previous Image")
                    next_btn = gr.Button("Next Image")
            with gr.Column(scale=1):
                bbox_display = gr.Textbox(
                    label="Bounding Boxes",
                    lines=10,
                    interactive=True,
                    placeholder="Draw boxes on the image..."
                )
                save_btn = gr.Button("Save Annotations")
                clear_btn = gr.Button("Clear Boxes")
                
                # Add prompt configuration
                gr.Markdown("### Prompt Configuration")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=3,
                    value=annotator.system_prompt
                )
                user_prompt = gr.Textbox(
                    label="User Prompt",
                    lines=3,
                    value=annotator.user_prompt
                )
                update_prompts_btn = gr.Button("Update Prompts")
                prompt_status = gr.Textbox(
                    label="Prompt Update Status",
                    interactive=False
                )
        
        image_input.select(
            fn=annotator.handle_select,
            inputs=[image_input],
            outputs=[image_input, bbox_display]
        )
        
        prev_btn.click(
            fn=lambda text: annotator.navigate_images('prev', text),
            inputs=[bbox_display],
            outputs=[image_input, bbox_display]
        )
        
        next_btn.click(
            fn=lambda text: annotator.navigate_images('next', text),
            inputs=[bbox_display],
            outputs=[image_input, bbox_display]
        )
        
        save_btn.click(
            fn=annotator.save_annotations,
            inputs=[bbox_display],
            outputs=gr.Textbox(label="Save Status")
        )
        
        clear_btn.click(
            fn=annotator.clear_boxes,
            outputs=[image_input, bbox_display]
        )
        
        # Add prompt update handler
        update_prompts_btn.click(
            fn=annotator.update_prompts,
            inputs=[system_prompt, user_prompt],
            outputs=prompt_status
        )
        
        if annotator.image_files:
            initial_image = f"data/input_images/{annotator.image_files[0]}"
            annotator.current_image_path = initial_image
            annotator.bbox_data = annotator.load_existing_annotations(initial_image)
            image_input.value = annotator.draw_boxes_on_image(initial_image)
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
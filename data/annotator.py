import gradio as gr
import os
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import anthropic
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import random

class ImageAnnotator:
    def __init__(self):
        self.screenshots_dir = Path("../tmp/data/screenshots")
        self.annotations_dir = Path("../tmp/data/annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_files = sorted(list(self.screenshots_dir.glob("*")))
        self.current_index = 0
        self.current_image_path = self.image_files[0]
        
        self.dot_radius = 5
        
        # Try to load a system font, fall back to default if not found
        try:
            # For Mac/Linux
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except OSError:
            try:
                # For Windows
                self.font = ImageFont.truetype("arial.ttf", 20)
            except OSError:
                # Fallback to default
                self.font = ImageFont.load_default()
        
        self.client = anthropic.Anthropic()
    
    def get_image_hash(self, image_path):
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_annotations(self, image_path):
        annotation_file = self.annotations_dir / f"{image_path.stem}.json"
        if annotation_file.exists():
            with open(annotation_file, "r") as f:
                return json.load(f)
        return {"elements": [], "image_name": image_path.name}
    
    def save_annotations(self, image_path: Path, annotations):
        img = Image.open(image_path)
        annotations["image_size"] = img.size
        annotations["image_hash"] = self.get_image_hash(image_path)
        
        annotation_file = self.annotations_dir / f"{image_path.stem}.json"
        with open(annotation_file, "w") as f:
            json.dump(annotations, f, indent=2)
    
    def get_recent_examples(self):
        """Get all name variations from the last 2 elements across all annotation files"""
        all_elements = []
        
        # Collect elements from all files
        for file in self.annotations_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if "elements" in data and data["elements"]:
                        # Add timestamp to sort by file modification time
                        file_time = file.stat().st_mtime
                        for element in data["elements"]:
                            if "names" in element:
                                all_elements.append({
                                    "time": file_time,
                                    "names": element["names"]
                                })
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
        # Sort by timestamp and get last 2 elements
        all_elements.sort(key=lambda x: x["time"], reverse=True)
        recent_elements = all_elements[:2]
        
        # Flatten all name variations from the last 2 elements
        recent_names = []
        for element in recent_elements:
            recent_names.extend(element["names"])
            
        return recent_names
    
    def get_element_description(self, original_image_path, annotated_image):
        """Get Claude's description of the clicked UI element"""
        try:
            # Get last 5 examples instead of random ones
            examples = self.get_recent_examples()
            
            # Create the base prompt
            base_prompt = """I'll show you two screenshots. The second one has a red dot indicating a UI element. 
Please provide THREE unique variations of names for this UI element, each on a new line. 
The names should describe the same element but with different levels of detail or focus.
For example:
- sign in
- sign in button
- red sign in button

The names should not use snake_case or camelCase. Focus on function and location. Be concise but specific."""
            
            # Only add examples if they exist
            if examples:
                prompt = f"{base_prompt}\n\nExamples of recent element names: {', '.join(examples)}.\n\nProvide exactly 3 variations, one per line:"
            else:
                prompt = f"{base_prompt}\n\nProvide exactly 3 variations, one per line:"
            
            # Convert original image to base64
            with Image.open(original_image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                orig_buffer = io.BytesIO()
                img.save(orig_buffer, format='PNG')
                orig_buffer.seek(0)
                orig_base64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')

            # Convert annotated image to base64
            if annotated_image.mode != 'RGB':
                annotated_image = annotated_image.convert('RGB')
            ann_buffer = io.BytesIO()
            annotated_image.save(ann_buffer, format='PNG')
            ann_buffer.seek(0)
            ann_base64 = base64.b64encode(ann_buffer.getvalue()).decode('utf-8')

            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": orig_base64
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": ann_base64
                            }
                        }
                    ]
                }]
            )
            
            response = message.content[0].text.strip()
            
            # Split response into lines and clean up
            variations = [line.strip('- ').strip() for line in response.split('\n') if line.strip()][:3]
            
            # Return the variations directly
            return variations
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return ["Error getting element description"]
    
    def draw_clicks_on_image(self, image_path, elements, include_ids=True):
        """Draw clicks on the image with optional IDs"""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        r = self.dot_radius
        
        for element in elements:
            if isinstance(element, list):  # Handle raw coordinate list
                x, y = element
            else:  # Handle dictionary format
                x, y = element["coordinates"]
                element_id = element.get("id", "")
            
            # Draw the red dot
            draw.ellipse([x-r, y-r, x+r, y+r], fill='red')
            
            # Draw only the ID if requested
            if include_ids and isinstance(element, dict) and element_id:
                label = f"{element_id}"  # Only show the ID number
                draw.text((x+r+5, y-10), label, font=self.font, fill='red')
        
        return img
    
    def update_click_coordinates(self, image_path, clicks_str, evt: gr.SelectData):
        # Load existing annotations first
        annotations = self.load_annotations(self.current_image_path)
        new_click = [evt.index[0], evt.index[1]]
        
        # Create image with just the new dot (no IDs) for Claude
        annotated_image_for_claude = self.draw_clicks_on_image(
            str(self.current_image_path), 
            [new_click], 
            include_ids=False
        )
        
        # Get element description variations from Claude
        variations = self.get_element_description(
            str(self.current_image_path),
            annotated_image_for_claude
        )
        
        # Update annotations with all name variations
        if "elements" not in annotations:
            annotations["elements"] = []
        
        new_id = len(annotations["elements"]) + 1
        annotations["elements"].append({
            "id": new_id,
            "coordinates": new_click,
            "names": variations
        })
        
        # Save annotations
        self.save_annotations(self.current_image_path, annotations)
        
        # Draw the updated image with all clicks and IDs
        updated_image = self.draw_clicks_on_image(str(self.current_image_path), annotations["elements"])
        
        return (updated_image,
                json.dumps(annotations["elements"], indent=2),
                self.format_clicks_for_display(annotations["elements"]),
                "✓ Added new element")
    
    def format_clicks_for_display(self, elements):
        if not elements:
            return "No clicks recorded"
        
        lines = []
        for element in elements:
            x, y = element["coordinates"]
            lines.append(f"ID {element['id']}: ({x}, {y})")
            if "names" in element:  # Handle both old and new format
                lines.append("Names:")
                for i, name in enumerate(element["names"], 1):
                    lines.append(f"{i}. {name}")
            else:
                lines.append(element.get("name", "Unnamed"))
            lines.append("---")
        return "\n".join(lines)
    
    def parse_display_text(self, text):
        """Parse the display text back into annotations structure"""
        lines = text.strip().split('\n')
        elements = []
        
        current_id = None
        current_coords = None
        current_names = []
        
        for line in lines:
            line = line.strip()
            if not line or line == "---":
                if current_id is not None and current_coords is not None:
                    elements.append({
                        "id": current_id,
                        "coordinates": current_coords,
                        "names": current_names if current_names else ["Unnamed"]
                    })
                current_id = None
                current_coords = None
                current_names = []
                continue
                
            if line.startswith("ID "):
                try:
                    id_part, coords_part = line.split(": ")
                    current_id = int(id_part.replace("ID ", ""))
                    coords_str = coords_part.strip("()").split(",")
                    current_coords = [float(coords_str[0]), float(coords_str[1])]
                except:
                    continue
            elif line == "Names:":
                continue
            elif line.startswith(("1.", "2.", "3.")):  # Handle numbered names
                name = line[3:].strip()  # Remove number and dot
                current_names.append(name)
        
        return {"elements": elements}
    
    def update_from_text(self, text):
        """Update annotations from edited text"""
        try:
            parsed = self.parse_display_text(text)
            annotations = self.load_annotations(self.current_image_path)
            annotations.update(parsed)
            self.save_annotations(self.current_image_path, annotations)
            
            # Redraw image with updated annotations
            updated_image = self.draw_clicks_on_image(str(self.current_image_path), annotations["elements"])
            
            return (updated_image,
                    json.dumps(annotations["elements"], indent=2),
                    "✓ Updated annotations")
        except Exception as e:
            print(f"Error updating annotations: {e}")
            return None, None, "Error updating annotations"
    
    def delete_annotation(self, index):
        """Delete an annotation by its index"""
        annotations = self.load_annotations(self.current_image_path)
        if "elements" in annotations and 0 <= index < len(annotations["elements"]):
            # Remove the element
            annotations["elements"].pop(index)
            # Reindex remaining elements
            for i, elem in enumerate(annotations["elements"], 1):
                elem["id"] = i
            
            self.save_annotations(self.current_image_path, annotations)
            
            # Redraw image with remaining annotations
            updated_image = self.draw_clicks_on_image(str(self.current_image_path), annotations["elements"])
            
            return (updated_image,
                    json.dumps(annotations["elements"], indent=2),
                    self.format_clicks_for_display(annotations["elements"]),
                    "✓ Annotation deleted")
    
    def edit_annotation(self, index, new_name):
        """Edit an annotation's name"""
        annotations = self.load_annotations(self.current_image_path)
        if "elements" in annotations and 0 <= index < len(annotations["elements"]):
            annotations["elements"][index]["name"] = new_name
            self.save_annotations(self.current_image_path, annotations)
            return self.format_clicks_for_display(annotations["elements"])
    
    def navigate_images(self, direction):
        if direction == "next":
            self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
        else:
            self.current_index = max(self.current_index - 1, 0)
            
        self.current_image_path = self.image_files[self.current_index]
        
        # Load existing annotations for the new image
        annotations = self.load_annotations(self.current_image_path)
        
        # Always draw the image with its annotations, even if empty
        current_image = self.draw_clicks_on_image(str(self.current_image_path), annotations["elements"])
        
        return (current_image, 
                json.dumps(annotations["elements"], indent=2),
                self.format_clicks_for_display(annotations["elements"]),
                f"Image {self.current_index + 1} of {len(self.image_files)}",
                "")  # Clear save status when navigating
    
    def save_click_coordinates(self, _, clicks_str):
        annotations = self.load_annotations(self.current_image_path)
        # We only need to save if there are no elements yet
        if "elements" not in annotations:
            annotations["elements"] = []
        self.save_annotations(self.current_image_path, annotations)
        return "✓ Saved annotation"

    def create_ui(self):
        initial_annotations = self.load_annotations(self.image_files[0])
        initial_image = self.draw_clicks_on_image(
            str(self.image_files[0]), 
            initial_annotations["elements"]
        ) if initial_annotations["elements"] else str(self.image_files[0])

        with gr.Blocks() as app:
            with gr.Row():
                with gr.Column(scale=4):
                    progress_text = gr.Markdown(
                        f"Image {self.current_index + 1} of {len(self.image_files)}"
                    )
                    image = gr.Image(
                        initial_image,
                        type="filepath",
                        interactive=True,
                        height=800
                    )
                    with gr.Row():
                        prev_btn = gr.Button("Previous")
                        next_btn = gr.Button("Next")
                
                with gr.Column(scale=1):
                    save_status = gr.Markdown("")
                    
                    clicks_json = gr.Textbox(
                        visible=False,
                        value=json.dumps(initial_annotations["elements"], indent=2)
                    )
                    clicks_display = gr.Textbox(
                        label="Click Coordinates and Descriptions (Edit directly)",
                        value=self.format_clicks_for_display(
                            initial_annotations["elements"]
                        ),
                        lines=15,
                        interactive=True
                    )
            
            save_btn = gr.Button("Save Changes")
            
            # Event handlers
            image.select(
                fn=self.update_click_coordinates,
                inputs=[image, clicks_json],
                outputs=[image, clicks_json, clicks_display, save_status]
            )
            
            clicks_display.change(
                fn=self.update_from_text,
                inputs=[clicks_display],
                outputs=[image, clicks_json, save_status]
            )
            
            prev_btn.click(
                fn=lambda: self.navigate_images("prev"),
                inputs=[],
                outputs=[image, clicks_json, clicks_display, progress_text, save_status]
            )
            
            next_btn.click(
                fn=lambda: self.navigate_images("next"),
                inputs=[],
                outputs=[image, clicks_json, clicks_display, progress_text, save_status]
            )
            
            save_btn.click(
                fn=self.save_click_coordinates,
                inputs=[image, clicks_json],
                outputs=[save_status]
            )
        
        return app

if __name__ == "__main__":
    annotator = ImageAnnotator()
    app = annotator.create_ui()
    app.launch()

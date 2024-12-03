import gradio as gr
import os
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

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
    
    def get_image_hash(self, image_path):
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_annotations(self, image_path):
        annotation_file = self.annotations_dir / f"{image_path.stem}.json"
        if annotation_file.exists():
            with open(annotation_file, "r") as f:
                return json.load(f)
        return {"clicks": [], "image_name": image_path.name}
    
    def save_annotations(self, image_path: Path, annotations):
        img = Image.open(image_path)
        annotations["image_size"] = img.size
        annotations["image_hash"] = self.get_image_hash(image_path)
        
        annotation_file = self.annotations_dir / f"{image_path.stem}.json"
        with open(annotation_file, "w") as f:
            json.dump(annotations, f, indent=2)
    
    def draw_clicks_on_image(self, image_path, clicks):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Draw each click as a red dot with its ID
        for i, (x, y) in enumerate(clicks, 1):
            # Draw the dot
            draw.ellipse(
                [(x - self.dot_radius, y - self.dot_radius), 
                 (x + self.dot_radius, y + self.dot_radius)],
                fill='red'
            )
            
            # Draw the ID number
            text = str(i)
            text_x = x + self.dot_radius + 5
            text_y = y - self.dot_radius - 5
            
            # Create thicker outline for better visibility
            outline_width = 2
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    draw.text(
                        (text_x + dx, text_y + dy), 
                        text, 
                        font=self.font,
                        fill='white'
                    )
            
            # Draw the text in red
            draw.text(
                (text_x, text_y), 
                text, 
                font=self.font,
                fill='red'
            )
        
        return img
    
    def update_click_coordinates(self, image_path, clicks_str, evt: gr.SelectData):
        clicks = json.loads(clicks_str) if clicks_str else []
        clicks.append([evt.index[0], evt.index[1]])
        
        # Save immediately after each click
        annotations = {"clicks": clicks, "image_name": self.current_image_path.name}
        self.save_annotations(self.current_image_path, annotations)
        
        # Draw the updated image with all clicks
        updated_image = self.draw_clicks_on_image(str(self.current_image_path), clicks)
        
        return (updated_image,
                json.dumps(clicks, indent=2),
                self.format_clicks_for_display(clicks),
                "✓ Saved annotation")
    
    def format_clicks_for_display(self, clicks):
        if not clicks:
            return "No clicks recorded"
        return "\n".join(f"ID {i+1}: ({x}, {y})" for i, (x, y) in enumerate(clicks))
    
    def navigate_images(self, direction):
        if direction == "next":
            self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
        else:
            self.current_index = max(self.current_index - 1, 0)
            
        self.current_image_path = self.image_files[self.current_index]
        
        # Load existing annotations for the new image
        annotations = self.load_annotations(self.current_image_path)
        
        # Always draw the image with its annotations, even if empty
        current_image = self.draw_clicks_on_image(str(self.current_image_path), annotations["clicks"])
        
        return (current_image, 
                json.dumps(annotations["clicks"], indent=2),
                self.format_clicks_for_display(annotations["clicks"]),
                f"Image {self.current_index + 1} of {len(self.image_files)}",
                "")  # Clear save status when navigating
    
    def save_click_coordinates(self, _, clicks_str):
        clicks = json.loads(clicks_str) if clicks_str else []
        annotations = {"clicks": clicks, "image_name": self.current_image_path.name}
        self.save_annotations(self.current_image_path, annotations)
        return "✓ Saved annotation"

    def create_ui(self):
        # Get initial image with annotations
        initial_annotations = self.load_annotations(self.image_files[0])
        initial_image = self.draw_clicks_on_image(
            str(self.image_files[0]), 
            initial_annotations["clicks"]
        ) if initial_annotations["clicks"] else str(self.image_files[0])

        with gr.Blocks() as app:
            with gr.Row():
                with gr.Column(scale=4):
                    # Add progress indicator
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
                    # Add save notification
                    save_status = gr.Markdown("")
                    
                    clicks_json = gr.Textbox(
                        visible=False,
                        value=json.dumps(initial_annotations["clicks"], indent=2)
                    )
                    clicks_display = gr.Textbox(
                        label="Click Coordinates",
                        value=self.format_clicks_for_display(initial_annotations["clicks"]),
                        lines=5,
                        interactive=False
                    )
            
            save_btn = gr.Button("Save Coordinates")
            
            # Updated event handlers
            image.select(
                fn=self.update_click_coordinates,
                inputs=[image, clicks_json],
                outputs=[image, clicks_json, clicks_display, save_status]
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

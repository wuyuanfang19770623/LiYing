import cv2
from PIL import Image, ImageDraw
import numpy as np

class PhotoSheetGenerator:
    def __init__(self, five_inch_size=(1050, 1500), dpi=300):
        self.five_inch_size = five_inch_size
        self.dpi = dpi

    @staticmethod
    def cv2_to_pillow(cv2_image):
        """Convert OpenCV image data to Pillow image"""
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image_rgb)

    @staticmethod
    def pillow_to_cv2(pillow_image):
        """Convert Pillow image to OpenCV image data"""
        cv2_image_rgb = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
        return cv2_image_rgb

    def generate_photo_sheet(self, one_inch_photo_cv2, rows=3, cols=3, rotate=False, add_crop_lines=True, auto_adjust=True):
        one_inch_height, one_inch_width = one_inch_photo_cv2.shape[:2]

        # Convert OpenCV image data to Pillow image
        one_inch_photo_pillow = self.cv2_to_pillow(one_inch_photo_cv2)

        # Rotate photo
        if rotate:
            one_inch_photo_pillow = one_inch_photo_pillow.rotate(90, expand=True)
            one_inch_height, one_inch_width = one_inch_width, one_inch_height

        # Create photo sheet (white background)
        five_inch_photo = Image.new('RGB', self.five_inch_size, 'white')

        # Calculate positions for the photos on the sheet
        total_width = cols * one_inch_width
        total_height = rows * one_inch_height

        if total_width > self.five_inch_size[0] or total_height > self.five_inch_size[1]:
            # Calculate maximum possible layout
            max_cols = self.five_inch_size[0] // one_inch_width
            max_rows = self.five_inch_size[1] // one_inch_height
            max_photos = max_cols * max_rows
            requested_photos = rows * cols
            
            if auto_adjust and max_photos > 0:
                # Auto-adjust to maximum possible layout
                print(f"Warning: Requested layout ({rows}x{cols} = {requested_photos} photos) exceeds sheet capacity. "
                      f"Auto-adjusting to maximum possible layout: {max_rows}x{max_cols} = {max_photos} photos.")
                rows = max_rows
                cols = max_cols
                # Recalculate with adjusted layout
                total_width = cols * one_inch_width
                total_height = rows * one_inch_height
            else:
                error_msg = (f"The specified layout ({rows}x{cols} = {requested_photos} photos) exceeds the size of the photo sheet. "
                            f"Photo size: {one_inch_width}x{one_inch_height}px, Sheet size: {self.five_inch_size[0]}x{self.five_inch_size[1]}px. "
                            f"Maximum possible layout: {max_rows}x{max_cols} = {max_photos} photos. "
                            f"Please reduce the number of rows/columns or use a larger sheet size.")
                raise ValueError(error_msg)

        start_x = (self.five_inch_size[0] - total_width) // 2
        start_y = (self.five_inch_size[1] - total_height) // 2

        # Arrange photos on the sheet in an n*m layout
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * one_inch_width
                y = start_y + i * one_inch_height
                five_inch_photo.paste(one_inch_photo_pillow, (x, y))

        # Draw crop lines if requested
        if add_crop_lines:
            draw = ImageDraw.Draw(five_inch_photo)
            
            # Draw outer rectangle
            draw.rectangle([start_x, start_y, start_x + total_width, start_y + total_height], outline="black")
            draw.rectangle([start_x, start_y, self.five_inch_size[0], self.five_inch_size[1]], outline="black")
            
            # Draw inner lines
            for i in range(1, rows):
                y = start_y + i * one_inch_height
                draw.line([(start_x, y), (start_x + total_width, y)], fill="black")
            
            for j in range(1, cols):
                x = start_x + j * one_inch_width
                draw.line([(x, start_y), (x, start_y + total_height)], fill="black")

        # Set the DPI information
        five_inch_photo.info['dpi'] = (self.dpi, self.dpi)

        # Return the generated photo sheet as a Pillow image
        return self.pillow_to_cv2(five_inch_photo)

    def save_photo_sheet(self, photo_sheet_cv, output_path):
        """Save the generated photo sheet as an image file"""
        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("output_path must be a valid image file path ending with .png, .jpg, or .jpeg")
        try:
            photo_sheet = self.cv2_to_pillow(photo_sheet_cv)
            photo_sheet.save(output_path, dpi=(self.dpi, self.dpi))
        except Exception as e:
            raise IOError(f"Failed to save photo: {e}")
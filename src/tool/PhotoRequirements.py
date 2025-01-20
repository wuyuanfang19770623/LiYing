import os

from ConfigManager import ConfigManager

class PhotoRequirements:
    def __init__(self, language=None, size_file=None, color_file=None):
        if language is None:
            language = os.getenv('CLI_LANGUAGE', 'zh')
        self.config_manager = ConfigManager(language, size_file, color_file)

    def get_requirements(self, photo_type):
        if not isinstance(photo_type, str):
            raise TypeError("Photo_type must be a string.")
        
        requirements = self.config_manager.get_size_config(photo_type)
        if requirements:
            return {
                'print_size': f"{requirements['PrintWidth']}cm x {requirements['PrintHeight']}cm" if requirements['PrintWidth'] and requirements['PrintHeight'] else 'N/A',
                'electronic_size': f"{requirements['ElectronicWidth']}px x {requirements['ElectronicHeight']}px",
                'resolution': f"{requirements['Resolution']}dpi" if requirements['Resolution'] else 'N/A',
                'file_format': requirements['FileFormat'],
                'file_size': f"{requirements['FileSizeMin']}-{requirements['FileSizeMax']}KB" if requirements['FileSizeMin'] and requirements['FileSizeMax'] else 'N/A'
            }
        else:
            return None

    def list_photo_types(self):
        return self.config_manager.list_size_configs()

    def get_resize_image_list(self, photo_type):
        requirements = self.config_manager.get_size_config(photo_type)
        if not requirements:
            raise ValueError(f"Photo type '{photo_type}' does not exist in size configurations.")
        
        # Get electronic size
        electronic_width = requirements['ElectronicWidth']
        electronic_height = requirements['ElectronicHeight']
        
        # Get print size and convert to inches if provided
        print_width_cm = requirements.get('PrintWidth')
        print_height_cm = requirements.get('PrintHeight')
        print_width_inch = print_width_cm / 2.54 if print_width_cm else None
        print_height_inch = print_height_cm / 2.54 if print_height_cm else None
        
        # Get resolution (DPI)
        resolution = requirements.get('Resolution', 300)  # Default to 300 DPI if not specified
        
        # Calculate pixel dimensions based on print size and resolution
        if print_width_inch and print_height_inch:
            calc_width = int(print_width_inch * resolution)
            calc_height = int(print_height_inch * resolution)
            
            # Use the larger of calculated or electronic size
            width = max(calc_width, electronic_width)
            height = max(calc_height, electronic_height)
        else:
            # If print size is not provided, use electronic size
            width = electronic_width
            height = electronic_height
        
        # Calculate actual print size based on final pixel dimensions
        actual_print_width_cm = (width / resolution) * 2.54
        actual_print_height_cm = (height / resolution) * 2.54
        
        return {
            'width': width,
            'height': height,
            'resolution': resolution,
            'electronic_size': f"{width}px x {height}px",
            'print_size': f"{actual_print_width_cm:.2f}cm x {actual_print_height_cm:.2f}cm"
        }
    
    def switch_language(self, new_language):
        self.config_manager.switch_language(new_language)

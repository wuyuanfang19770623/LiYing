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

        width = requirements['ElectronicWidth']
        height = requirements['ElectronicHeight']
        electronic_size = f"{width}px x {height}px"
        
        return [width, height, electronic_size]
    
    def switch_language(self, new_language):
        self.config_manager.switch_language(new_language)

import unittest
import os
import sys
import cv2
import numpy as np
import json
import warnings

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'tool'))

from src.tool.ImageProcessor import ImageProcessor
from src.tool.PhotoSheetGenerator import PhotoSheetGenerator
from src.tool.PhotoRequirements import PhotoRequirements
from src.tool.ConfigManager import ConfigManager

class TestLiYing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_image_path = os.path.join(project_root, 'images', 'test1.jpg')
        cls.output_dir = os.path.join(project_root, 'tests', 'output')
        os.makedirs(cls.output_dir, exist_ok=True)

        # Set model paths
        cls.yolov8_model_path = os.path.join(project_root, 'src', 'model', 'yolov8n-pose.onnx')
        cls.yunet_model_path = os.path.join(project_root, 'src', 'model', 'face_detection_yunet_2023mar.onnx')
        cls.rmbg_model_path = os.path.join(project_root, 'src', 'model', 'RMBG-1.4-model.onnx')

    def get_first_valid_photo_size(self, photo_requirements):
        photo_types = photo_requirements.list_photo_types()
        return photo_types[0] if photo_types else None

    def check_models_exist(self):
        missing_models = []
        if not os.path.exists(self.yolov8_model_path):
            missing_models.append("YOLOv8")
        if not os.path.exists(self.yunet_model_path):
            missing_models.append("YuNet")
        if not os.path.exists(self.rmbg_model_path):
            missing_models.append("RMBG")
        return missing_models

    def test_image_processor(self):
        missing_models = self.check_models_exist()
        if missing_models:
            warnings.warn(f"Skipping image processing test: Missing model files {', '.join(missing_models)}")
            return

        photo_requirements = PhotoRequirements(language='en')
        
        processor = ImageProcessor(self.test_image_path, 
                                yolov8_model_path=self.yolov8_model_path,
                                yunet_model_path=self.yunet_model_path,
                                RMBG_model_path=self.rmbg_model_path)
        processor.photo_requirements_detector = photo_requirements
        
        # Test crop and correct
        processor.crop_and_correct_image()
        self.assertIsNotNone(processor.photo.image)
        
        # Test background change
        processor.change_background([255, 0, 0])  # Red background, using list
        self.assertIsNotNone(processor.photo.image)
        
        # Test resize
        first_photo_size = self.get_first_valid_photo_size(photo_requirements)
        self.assertIsNotNone(first_photo_size, "No valid photo size found")
        processor.resize_image(first_photo_size)
        self.assertIsNotNone(processor.photo.image)
        
        # Save processed image
        output_path = os.path.join(self.output_dir, 'processed_image.jpg')
        processor.save_photos(output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_photo_sheet_generator(self):
        missing_models = self.check_models_exist()
        if missing_models:
            warnings.warn(f"Skipping photo sheet generation test: Missing model files {', '.join(missing_models)}")
            return

        photo_requirements = PhotoRequirements(language='en')
        
        processor = ImageProcessor(self.test_image_path,
                                yolov8_model_path=self.yolov8_model_path,
                                yunet_model_path=self.yunet_model_path,
                                RMBG_model_path=self.rmbg_model_path)
        processor.photo_requirements_detector = photo_requirements
        processor.crop_and_correct_image()
        
        first_photo_size = self.get_first_valid_photo_size(photo_requirements)
        self.assertIsNotNone(first_photo_size, "No valid photo size found")
        processor.resize_image(first_photo_size)
        
        # Use size from configuration
        sheet_sizes = photo_requirements.config_manager.get_sheet_sizes()
        first_sheet_size = next(iter(sheet_sizes))
        sheet_config = photo_requirements.config_manager.get_size_config(first_sheet_size)
        generator = PhotoSheetGenerator((sheet_config['ElectronicWidth'], sheet_config['ElectronicHeight']))
        sheet = generator.generate_photo_sheet(processor.photo.image, 2, 2)
        
        self.assertIsNotNone(sheet)
        self.assertEqual(sheet.shape[:2], (sheet_config['ElectronicHeight'], sheet_config['ElectronicWidth']))
        
        output_path = os.path.join(self.output_dir, 'photo_sheet.jpg')
        generator.save_photo_sheet(sheet, output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_photo_requirements(self):
        requirements = PhotoRequirements(language='en')
        first_photo_size = self.get_first_valid_photo_size(requirements)
        self.assertIsNotNone(first_photo_size, "No valid photo size found")
        
        # Test getting resize image list
        size_list = requirements.get_resize_image_list(first_photo_size)
        self.assertEqual(len(size_list), 3)
        self.assertIsInstance(size_list[0], int)
        self.assertIsInstance(size_list[1], int)
        self.assertIsInstance(size_list[2], str)

def test_config_manager(self):
    # Test Chinese configuration
    config_manager_zh = ConfigManager(language='zh')
    config_manager_zh.load_configs()
    
    # Test getting photo size configurations
    photo_size_configs_zh = config_manager_zh.get_photo_size_configs()
    self.assertIsInstance(photo_size_configs_zh, dict)
    self.assertGreater(len(photo_size_configs_zh), 0)
    
    # Test getting photo sheet size configurations
    sheet_size_configs_zh = config_manager_zh.get_sheet_size_configs()
    self.assertIsInstance(sheet_size_configs_zh, dict)
    self.assertGreater(len(sheet_size_configs_zh), 0)

    # Test English configuration
    config_manager_en = ConfigManager(language='en')
    config_manager_en.load_configs()
    
    # Test getting photo size configurations
    photo_size_configs_en = config_manager_en.get_photo_size_configs()
    self.assertIsInstance(photo_size_configs_en, dict)
    self.assertGreater(len(photo_size_configs_en), 0)
    
    # Test getting photo sheet size configurations
    sheet_size_configs_en = config_manager_en.get_sheet_size_configs()
    self.assertIsInstance(sheet_size_configs_en, dict)
    self.assertGreater(len(sheet_size_configs_en), 0)

    # Check if key sets are the same
    self.assertEqual(set(photo_size_configs_zh.keys()), set(photo_size_configs_en.keys()))

    # Check if some common sizes exist
    common_sizes_zh = ['一寸', '二寸', '五寸', '六寸']
    common_sizes_en = ['one_inch', 'two_inch', 'five_inch', 'six_inch']
    
    for size_zh, size_en in zip(common_sizes_zh, common_sizes_en):
        self.assertIn(size_zh, photo_size_configs_zh, f"Chinese config should contain {size_zh}")
        self.assertIn(size_en, photo_size_configs_en, f"English config should contain {size_en}")

    # Test language switching functionality
    config_manager = ConfigManager(language='zh')
    self.assertEqual(config_manager.language, 'zh')
    config_manager.switch_language('en')
    self.assertEqual(config_manager.language, 'en')

    # Test if configuration file paths are correctly updated
    self.assertTrue(config_manager.size_file.endswith('size_en.csv'))
    self.assertTrue(config_manager.color_file.endswith('color_en.csv'))

    # Test i18n JSON files
    i18n_dir = os.path.join(project_root, 'src', 'webui', 'i18n')
    
    with open(os.path.join(i18n_dir, 'en.json'), 'r', encoding='utf-8') as f:
        en_i18n = json.load(f)
    
    with open(os.path.join(i18n_dir, 'zh.json'), 'r', encoding='utf-8') as f:
        zh_i18n = json.load(f)
    
    # Check if i18n files have the same keys
    self.assertEqual(set(en_i18n.keys()), set(zh_i18n.keys()))
    
    # Check if at least one value is different
    different_values = [key for key in en_i18n.keys() if en_i18n[key] != zh_i18n[key]]
    self.assertGreater(len(different_values), 0, "English and Chinese i18n files should have different translations")


if __name__ == '__main__':
    unittest.main()

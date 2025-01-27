import unittest
import csv
import os
import sys

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'tool'))

def inch_to_cm(inch):
    return inch * 2.54

def cm_to_inch(cm):
    return cm / 2.54

def calculate_electronic_size(print_width, print_height, resolution):
    electronic_width = round(print_width / inch_to_cm(1) * resolution)
    electronic_height = round(print_height / inch_to_cm(1) * resolution)
    return electronic_width, electronic_height

class TestPhotoSizes(unittest.TestCase):
    def setUp(self):
        self.csv_path = os.path.join('..\data', 'size_zh.csv')
        self.tolerance_pixels = 1
        self.tolerance_cm = 0.1

    def test_photo_sizes(self):
        with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                with self.subTest(name=row['Name']):
                    self.validate_row(row)

    def validate_row(self, row):
        name = row['Name']
        print_width = float(row['PrintWidth']) if row['PrintWidth'] else None
        print_height = float(row['PrintHeight']) if row['PrintHeight'] else None
        electronic_width = int(row['ElectronicWidth']) if row['ElectronicWidth'] else None
        electronic_height = int(row['ElectronicHeight']) if row['ElectronicHeight'] else None
        resolution = int(row['Resolution']) if row['Resolution'] else 300

        if print_width and print_height and electronic_width and electronic_height:
            self.validate_electronic_size(name, print_width, print_height, electronic_width, electronic_height, resolution)
        elif electronic_width and electronic_height:
            self.validate_print_size(name, electronic_width, electronic_height, print_width, print_height, resolution)
        else:
            self.fail(f"Insufficient data for {name}")

    def validate_electronic_size(self, name, print_width, print_height, electronic_width, electronic_height, resolution):
        calculated_width, calculated_height = calculate_electronic_size(print_width, print_height, resolution)
        
        width_diff = abs(calculated_width - electronic_width)
        height_diff = abs(calculated_height - electronic_height)
        
        self.assertLessEqual(width_diff, self.tolerance_pixels, 
                             f"Width mismatch for {name}: calculated {calculated_width}, actual {electronic_width}")
        self.assertLessEqual(height_diff, self.tolerance_pixels, 
                             f"Height mismatch for {name}: calculated {calculated_height}, actual {electronic_height}")

    def validate_print_size(self, name, electronic_width, electronic_height, print_width, print_height, resolution):
        calculated_print_width = round(cm_to_inch(electronic_width / resolution * inch_to_cm(1)), 1)
        calculated_print_height = round(cm_to_inch(electronic_height / resolution * inch_to_cm(1)), 1)
        
        if print_width and print_height:
            width_diff = abs(calculated_print_width - print_width)
            height_diff = abs(calculated_print_height - print_height)
            
            self.assertLessEqual(width_diff, self.tolerance_cm, 
                                 f"Print width mismatch for {name}: calculated {calculated_print_width}, actual {print_width}")
            self.assertLessEqual(height_diff, self.tolerance_cm, 
                                 f"Print height mismatch for {name}: calculated {calculated_print_height}, actual {print_height}")
        else:
            self.fail(f"Missing print size for {name}: Calculated print size: {calculated_print_width}x{calculated_print_height}")

if __name__ == '__main__':
    unittest.main()

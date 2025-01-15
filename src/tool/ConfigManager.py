import csv
import os
from typing import Dict, List, Optional, Tuple

class ConfigManager:
    _instance = None

    def __new__(cls, language: str = 'zh', size_file: Optional[str] = None, color_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, language: str = 'zh', size_file: Optional[str] = None, color_file: Optional[str] = None):
        if not hasattr(self, 'initialized'):
            self.language = language
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            default_size_file = os.path.join(project_root, 'data', f'size_{language}.csv')
            default_color_file = os.path.join(project_root, 'data', f'color_{language}.csv')
            self.size_file = os.path.abspath(size_file) if size_file else default_size_file
            self.color_file = os.path.abspath(color_file) if color_file else default_color_file
            self.size_config: Dict[str, Dict] = {}
            self.color_config: Dict[str, Dict] = {}
            self.load_configs()
            self.initialized = True

    def load_configs(self):
        """Load both size and color configurations."""
        self.size_config.clear()
        self.color_config.clear()
        self.load_size_config()
        self.load_color_config()

    def load_size_config(self):
        """Load size configuration from CSV file."""
        if not os.path.exists(self.size_file):
            raise FileNotFoundError(f"Size configuration file {self.size_file} not found.")

        with open(self.size_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.size_config[row['Name']] = {
                    'PrintWidth': float(row['PrintWidth']) if row['PrintWidth'] else None,
                    'PrintHeight': float(row['PrintHeight']) if row['PrintHeight'] else None,
                    'ElectronicWidth': int(row['ElectronicWidth']),
                    'ElectronicHeight': int(row['ElectronicHeight']),
                    'Resolution': int(row['Resolution']) if row['Resolution'] else None,
                    'FileFormat': row['FileFormat'],
                    'FileSizeMin': int(row['FileSizeMin']) if row['FileSizeMin'] else None,
                    'FileSizeMax': int(row['FileSizeMax']) if row['FileSizeMax'] else None,
                    'Type': row['Type'],
                    'Notes': row['Notes']
                }

    def load_color_config(self):
        """Load color configuration from CSV file."""
        if not os.path.exists(self.color_file):
            raise FileNotFoundError(f"Color configuration file {self.color_file} not found.")

        with open(self.color_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.color_config[row['Name']] = {
                    'R': int(row['R']),
                    'G': int(row['G']),
                    'B': int(row['B']),
                    'Notes': row['Notes']
                }

    def get_size_config(self, name: str) -> Optional[Dict]:
        """Get size configuration by name."""
        return self.size_config.get(name)

    def get_photo_size_configs(self) -> Dict[str, Dict]:
        """Get all photo size configurations."""
        return {name: config for name, config in self.size_config.items() if config['Type'] in ['photo', 'both']}

    def get_sheet_size_configs(self) -> Dict[str, Dict]:
        """Get all sheet size configurations."""
        return {name: config for name, config in self.size_config.items() if config['Type'] in ['sheet', 'both']}

    def get_color_config(self, name: str) -> Optional[Dict]:
        """Get color configuration by name."""
        return self.color_config.get(name)

    def add_size_config(self, name: str, config: Dict):
        """Add a new size configuration."""
        self.size_config[name] = config
        self.save_size_config()

    def add_color_config(self, name: str, config: Dict):
        """Add a new color configuration."""
        self.color_config[name] = config
        self.save_color_config()

    def update_size_config(self, name: str, config: Dict):
        """Update an existing size configuration."""
        if name in self.size_config:
            self.size_config[name].update(config)
            self.save_size_config()
        else:
            raise KeyError(f"Size configuration '{name}' not found.")

    def update_color_config(self, name: str, config: Dict):
        """Update an existing color configuration."""
        if name in self.color_config:
            self.color_config[name].update(config)
            self.save_color_config()
        else:
            raise KeyError(f"Color configuration '{name}' not found.")

    def delete_size_config(self, name: str):
        """Delete a size configuration."""
        if name in self.size_config:
            del self.size_config[name]
            self.save_size_config()
        else:
            raise KeyError(f"Size configuration '{name}' not found.")

    def delete_color_config(self, name: str):
        """Delete a color configuration."""
        if name in self.color_config:
            del self.color_config[name]
            self.save_color_config()
        else:
            raise KeyError(f"Color configuration '{name}' not found.")

    def save_size_config(self):
        """Save size configuration to CSV file."""
        with open(self.size_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'PrintWidth', 'PrintHeight', 'ElectronicWidth', 'ElectronicHeight', 'Resolution', 'FileFormat', 'FileSizeMin', 'FileSizeMax', 'Type', 'Notes'])
            writer.writeheader()
            for name, config in self.size_config.items():
                writer.writerow({'Name': name, **config})

    def save_color_config(self):
        """Save color configuration to CSV file."""
        with open(self.color_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'R', 'G', 'B', 'Notes'])
            writer.writeheader()
            for name, config in self.color_config.items():
                writer.writerow({'Name': name, **config})

    def check_size_config_integrity(self, config: Dict) -> Tuple[bool, str]:
        """Check integrity of a size configuration."""
        required_fields = ['ElectronicWidth', 'ElectronicHeight', 'Type']
        for field in required_fields:
            if field not in config or config[field] is None:
                return False, f"Missing required field: {field}"
        return True, "Config is valid"

    def check_color_config_integrity(self, config: Dict) -> Tuple[bool, str]:
        """Check integrity of a color configuration."""
        required_fields = ['R', 'G', 'B']
        for field in required_fields:
            if field not in config or config[field] is None:
                return False, f"Missing required field: {field}"
        return True, "Config is valid"

    def switch_language(self, new_language: str):
        """Switch language and reload configurations."""
        self.language = new_language
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.size_file = os.path.join(project_root, 'data', f'size_{new_language}.csv')
        self.color_file = os.path.join(project_root, 'data', f'color_{new_language}.csv')
        try:
            self.load_configs()
        except FileNotFoundError as e:
            print(f"Error loading configuration files: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_photo_sizes(self) -> Dict[str, Dict]:
        """Get all photo sizes."""
        return {name: config for name, config in self.size_config.items() if config['Type'] in ['photo', 'both']}

    def get_sheet_sizes(self) -> Dict[str, Dict]:
        """Get all sheet sizes."""
        return {name: config for name, config in self.size_config.items() if config['Type'] in ['sheet', 'both']}

    def list_size_configs(self) -> List[str]:
        """List all size configuration names."""
        return list(self.size_config.keys())

    def list_color_configs(self) -> List[str]:
        """List all color configuration names."""
        return list(self.color_config.keys())

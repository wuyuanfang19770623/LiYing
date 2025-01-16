import gradio as gr
import os
import sys
import re
import cv2
import locale
import json
import argparse
import pandas as pd
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)

PROJECT_ROOT = project_root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TOOL_DIR = os.path.join(src_dir, 'tool')
MODEL_DIR = os.path.join(src_dir, 'model')

DEFAULT_YOLOV8_PATH = os.path.join(MODEL_DIR, 'yolov8n-pose.onnx')
DEFAULT_YUNET_PATH = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
DEFAULT_RMBG_PATH = os.path.join(MODEL_DIR, 'RMBG-1.4-model.onnx')
DEFAULT_SIZE_CONFIG = os.path.join(DATA_DIR, 'size_{}.csv')
DEFAULT_COLOR_CONFIG = os.path.join(DATA_DIR, 'color_{}.csv')

sys.path.extend([DATA_DIR, MODEL_DIR, TOOL_DIR])

from tool.ImageProcessor import ImageProcessor
from tool.PhotoSheetGenerator import PhotoSheetGenerator
from tool.PhotoRequirements import PhotoRequirements
from tool.ConfigManager import ConfigManager

def get_language():
    """Get the system language or default to English."""
    try:
        system_lang = locale.getlocale()[0].split('_')[0]
        return system_lang if system_lang in ['en', 'zh'] else 'en'
    except:
        return 'en'

def load_i18n_texts():
    """Load internationalization texts from JSON files."""
    i18n_dir = os.path.join(os.path.dirname(__file__), 'i18n')
    texts = {}
    for lang in ['en', 'zh']:
        with open(os.path.join(i18n_dir, f'{lang}.json'), 'r', encoding='utf-8') as f:
            texts[lang] = json.load(f)
    return texts

TEXTS = load_i18n_texts()

def t(key, language):
    """Translate a key to the specified language."""
    return TEXTS[language][key]

def parse_color(color_string):
    """Parse color string to RGB list."""
    if color_string is None:
        return [255, 255, 255]
    if color_string.startswith('#'):
        return [int(color_string.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    rgb_match = re.match(r'rgba?$(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)(?:,\s*[\d.]+)?$', color_string)
    if rgb_match:
        return [min(255, max(0, int(float(x)))) for x in rgb_match.groups()]
    return [255, 255, 255]

def process_image(img_path, yolov8_path, yunet_path, rmbg_path, photo_requirements, photo_type, photo_sheet_size, rgb_list, compress=False, change_background=False, rotate=False, resize=True, sheet_rows=3, sheet_cols=3, add_crop_lines=True):
    """Process the image with specified parameters."""
    processor = ImageProcessor(img_path, 
                            yolov8_model_path=yolov8_path,
                            yunet_model_path=yunet_path,
                            RMBG_model_path=rmbg_path,
                            rgb_list=rgb_list, 
                            y_b=compress)

    processor.crop_and_correct_image()

    if change_background:
        processor.change_background()

    if resize:
        processor.resize_image(photo_type)

    sheet_width, sheet_height, _ = photo_requirements.get_resize_image_list(photo_sheet_size)
    generator = PhotoSheetGenerator([sheet_width, sheet_height])
    photo_sheet_cv = generator.generate_photo_sheet(processor.photo.image, sheet_rows, sheet_cols, rotate, add_crop_lines)

    return {
        'final_image': photo_sheet_cv,
        'corrected_image': processor.photo.image,
    }

def create_demo(initial_language):
    """Create the Gradio demo interface."""
    config_manager = ConfigManager(language=initial_language)
    config_manager.load_configs()
    photo_requirements = PhotoRequirements(language=initial_language)

    def update_configs():
        nonlocal photo_size_configs, sheet_size_configs, color_configs, photo_size_choices, sheet_size_choices, color_choices
        photo_size_configs = config_manager.get_photo_size_configs()
        sheet_size_configs = config_manager.get_sheet_size_configs()
        color_configs = config_manager.color_config
        photo_size_choices = list(photo_size_configs.keys())
        sheet_size_choices = list(sheet_size_configs.keys())
        color_choices = [t('custom_color', config_manager.language)] + list(color_configs.keys())

    update_configs()

    photo_size_configs = config_manager.get_photo_size_configs()
    sheet_size_configs = config_manager.get_sheet_size_configs()
    color_configs = config_manager.color_config

    photo_size_choices = list(photo_size_configs.keys())
    sheet_size_choices = list(sheet_size_configs.keys())
    color_choices = [t('custom_color', initial_language)] + list(color_configs.keys())

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        language = gr.State(initial_language)

        title = gr.Markdown(f"# {t('title', initial_language)}")

        color_change_source = {"source": "custom"}

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label=t('upload_photo', initial_language), height=400)
                lang_dropdown = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh")], 
                    value=initial_language, 
                    label=t('language', initial_language)
                )

                with gr.Tabs() as tabs:
                    with gr.TabItem(t('key_param', initial_language)) as key_param_tab:
                        photo_type = gr.Dropdown(choices=photo_size_choices, label=t('photo_type', initial_language))
                        photo_sheet_size = gr.Dropdown(choices=sheet_size_choices, label=t('photo_sheet_size', initial_language))
                        with gr.Row():
                            preset_color = gr.Dropdown(choices=color_choices, label=t('preset_color', initial_language), value=t('custom_color', initial_language))
                            background_color = gr.ColorPicker(label=t('background_color', initial_language), value="#FFFFFF")
                        layout_only = gr.Checkbox(label=t('layout_only', initial_language), value=False)
                        sheet_rows = gr.Slider(minimum=1, maximum=10, step=1, value=3, label=t('sheet_rows', initial_language))
                        sheet_cols = gr.Slider(minimum=1, maximum=10, step=1, value=3, label=t('sheet_cols', initial_language))
                    
                    with gr.TabItem(t('advanced_settings', initial_language)) as advanced_settings_tab:
                        yolov8_path = gr.Textbox(label=t('yolov8_path', initial_language), value=DEFAULT_YOLOV8_PATH)
                        yunet_path = gr.Textbox(label=t('yunet_path', initial_language), value=DEFAULT_YUNET_PATH)
                        rmbg_path = gr.Textbox(label=t('rmbg_path', initial_language), value=DEFAULT_RMBG_PATH)
                        size_config = gr.Textbox(label=t('size_config', initial_language), value=DEFAULT_SIZE_CONFIG.format(initial_language))
                        color_config = gr.Textbox(label=t('color_config', initial_language), value=DEFAULT_COLOR_CONFIG.format(initial_language))
                        compress = gr.Checkbox(label=t('compress', initial_language), value=True)
                        change_background = gr.Checkbox(label=t('change_background', initial_language), value=True)
                        rotate = gr.Checkbox(label=t('rotate', initial_language), value=False)
                        resize = gr.Checkbox(label=t('resize', initial_language), value=True)
                        add_crop_lines = gr.Checkbox(label=t('add_crop_lines', initial_language), value=True)
                        confirm_advanced_settings = gr.Button(t('confirm_settings', initial_language))

                    with gr.TabItem(t('config_management', initial_language)) as config_management_tab:
                        with gr.Tabs() as config_tabs:
                            with gr.TabItem(t('size_config', initial_language)) as size_config_tab:
                                size_df = gr.Dataframe(
                                    value=pd.DataFrame(
                                        [
                                            [name] + list(config.values())
                                            for name, config in config_manager.size_config.items()
                                        ],
                                        columns=['Name'] + (list(next(iter(config_manager.size_config.values())).keys()) if config_manager.size_config else [])
                                    ),
                                    interactive=True,
                                    label=t('size_config_table', initial_language)
                                )
                                with gr.Row():
                                    add_size_btn = gr.Button(t('add_size', initial_language))
                                    update_size_btn = gr.Button(t('save_size', initial_language))

                            with gr.TabItem(t('color_config', initial_language)) as color_config_tab:
                                color_df = gr.Dataframe(
                                    value=pd.DataFrame(
                                        [
                                            [name] + list(config.values())
                                            for name, config in config_manager.color_config.items()
                                        ],
                                        columns=['Name', 'R', 'G', 'B', 'Notes']
                                    ),
                                    interactive=True,
                                    label=t('color_config_table', initial_language)
                                )
                                with gr.Row():
                                    add_color_btn = gr.Button(t('add_color', initial_language))
                                    update_color_btn = gr.Button(t('save_color', initial_language))

                        config_notification = gr.Textbox(label=t('config_notification', initial_language))

                process_btn = gr.Button(t('process_btn', initial_language))

            with gr.Column(scale=1):
                with gr.Tabs() as result_tabs:
                    with gr.TabItem(t('result', initial_language)) as result_tab:
                        output_image = gr.Image(label=t('final_image', initial_language), height=800)
                    with gr.TabItem(t('corrected_image', initial_language)) as corrected_image_tab:
                        corrected_output = gr.Image(label=t('corrected_image', initial_language), height=800)
                
                notification = gr.Textbox(label=t('notification', initial_language))

        def update_language(lang):
            """Update UI language and reload configs."""
            nonlocal config_manager, photo_requirements
            config_manager.switch_language(lang)
            photo_requirements.switch_language(lang)
            update_configs()
            
            new_photo_size_configs = config_manager.get_photo_size_configs()
            new_sheet_size_configs = config_manager.get_sheet_size_configs()
            color_configs = config_manager.color_config
            
            new_photo_size_choices = list(new_photo_size_configs.keys())
            new_sheet_size_choices = list(new_sheet_size_configs.keys())
            new_color_choices = [t('custom_color', lang)] + list(color_configs.keys())

            return {
                title: gr.update(value=f"# {t('title', lang)}"),
                input_image: gr.update(label=t('upload_photo', lang)),
                lang_dropdown: gr.update(label=t('language', lang)),
                photo_type: gr.update(choices=new_photo_size_choices, label=t('photo_type', lang), value=new_photo_size_choices[0] if new_photo_size_choices else None),
                photo_sheet_size: gr.update(choices=new_sheet_size_choices, label=t('photo_sheet_size', lang), value=new_sheet_size_choices[0] if new_sheet_size_choices else None),
                preset_color: gr.update(choices=new_color_choices, label=t('preset_color', lang)),
                background_color: gr.update(label=t('background_color', lang)),
                layout_only: gr.update(label=t('layout_only', lang)),
                sheet_rows: gr.update(label=t('sheet_rows', lang)),
                sheet_cols: gr.update(label=t('sheet_cols', lang)),
                yolov8_path: gr.update(label=t('yolov8_path', lang)),
                yunet_path: gr.update(label=t('yunet_path', lang)),
                rmbg_path: gr.update(label=t('rmbg_path', lang)),
                size_config: gr.update(label=t('size_config', lang), value=DEFAULT_SIZE_CONFIG.format(lang)),
                color_config: gr.update(label=t('color_config', lang), value=DEFAULT_COLOR_CONFIG.format(lang)),
                compress: gr.update(label=t('compress', lang)),
                change_background: gr.update(label=t('change_background', lang)),
                rotate: gr.update(label=t('rotate', lang)),
                resize: gr.update(label=t('resize', lang)),
                add_crop_lines: gr.update(label=t('add_crop_lines', lang)),
                process_btn: gr.update(value=t('process_btn', lang)),
                output_image: gr.update(label=t('final_image', lang)),
                corrected_output: gr.update(label=t('corrected_image', lang)),
                notification: gr.update(label=t('notification', lang)),
                key_param_tab: gr.update(label=t('key_param', lang)),
                advanced_settings_tab: gr.update(label=t('advanced_settings', lang)),
                config_management_tab: gr.update(label=t('config_management', lang)),
                size_config_tab: gr.update(label=t('size_config', lang)),
                color_config_tab: gr.update(label=t('color_config', lang)),
                confirm_advanced_settings: gr.update(value=t('confirm_settings', lang)),
                result_tab: gr.update(label=t('result', lang)),
                corrected_image_tab: gr.update(label=t('corrected_image', lang)),
                size_df: gr.update(
                    value=pd.DataFrame(
                        [[name] + list(config.values()) for name, config in config_manager.size_config.items()],
                        columns=['Name'] + (list(next(iter(config_manager.size_config.values())).keys()) if config_manager.size_config else [])
                    ),
                    label=t('size_config_table', lang)
                ),
                color_df: gr.update(
                    value=pd.DataFrame(
                        [[name] + list(config.values()) for name, config in config_manager.color_config.items()],
                        columns=['Name', 'R', 'G', 'B', 'Notes']
                    ),
                    label=t('color_config_table', lang)
                ),
                add_size_btn: gr.update(value=t('add_size', lang)),
                update_size_btn: gr.update(value=t('save_size', lang)),
                add_color_btn: gr.update(value=t('add_color', lang)),
                update_color_btn: gr.update(value=t('save_color', lang)),
                config_notification: gr.update(label=t('config_notification', lang)),
            }

        def confirm_advanced_settings_fn(yolov8_path, yunet_path, rmbg_path, size_config, color_config):
            config_manager.size_file = size_config
            config_manager.color_file = color_config
            config_manager.load_configs()
            update_configs()
            return {
                size_df: gr.update(value=pd.DataFrame(
                    [[name] + list(config.values()) for name, config in config_manager.size_config.items()],
                    columns=['Name'] + list(next(iter(config_manager.size_config.values())).keys())
                )),
                color_df: gr.update(value=pd.DataFrame(
                    [[name] + list(config.values()) for name, config in config_manager.color_config.items()],
                    columns=['Name', 'R', 'G', 'B', 'Notes']
                )),
                photo_type: gr.update(choices=photo_size_choices),
                photo_sheet_size: gr.update(choices=sheet_size_choices),
                preset_color: gr.update(choices=color_choices),
            }

        def update_background_color(preset, lang):
            """Update background color based on preset selection."""
            custom_color = t('custom_color', lang)
            if preset == custom_color:
                color_change_source["source"] = "custom"
                return gr.update()
            
            if preset in color_configs:
                color = color_configs[preset]
                hex_color = f"#{color['R']:02x}{color['G']:02x}{color['B']:02x}"
                color_change_source["source"] = "preset"
                return gr.update(value=hex_color)
            
            color_change_source["source"] = "custom"
            return gr.update(value="#FFFFFF")

        def update_preset_color(color, lang):
            """Update preset color dropdown based on color picker changes."""
            if color_change_source["source"] == "preset":
                color_change_source["source"] = "custom"
                return gr.update()
            custom_color = t('custom_color', lang)
            return gr.update(value=custom_color)

        def process_and_display(image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, photo_type, photo_sheet_size, background_color, compress, change_background, rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines):
            """Process and display the image with given parameters."""
            rgb_list = parse_color(background_color)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            temp_image_path = "temp_input_image.jpg"
            cv2.imwrite(temp_image_path, image_bgr)

            result = process_image(
                temp_image_path,
                yolov8_path,
                yunet_path,
                rmbg_path,
                photo_requirements,
                photo_type=photo_type,
                photo_sheet_size=photo_sheet_size,
                rgb_list=rgb_list,
                compress=compress,
                change_background=change_background and not layout_only,
                rotate=rotate,
                resize=resize,
                sheet_rows=sheet_rows,
                sheet_cols=sheet_cols,
                add_crop_lines=add_crop_lines
            )
            
            os.remove(temp_image_path)
            final_image_rgb = cv2.cvtColor(result['final_image'], cv2.COLOR_BGR2RGB)
            corrected_image_rgb = cv2.cvtColor(result['corrected_image'], cv2.COLOR_BGR2RGB)
            
            return final_image_rgb, corrected_image_rgb

        def add_size_config(df):
            """Add a new empty row to the size configuration table."""
            new_row = pd.DataFrame([['' for _ in df.columns]], columns=df.columns)
            updated_df = pd.concat([df, new_row], ignore_index=True)
            return updated_df, t('size_config_row_added', config_manager.language)

        def update_size_config(df):
            """Save all changes made to the size configuration table and remove empty rows."""
            updated_config = {}
            for _, row in df.iterrows():
                name = row['Name']
                if name and not row.iloc[1:].isna().all():  # Check if name exists and not all other fields are empty
                    config = row.to_dict()
                    config.pop('Name')
                    updated_config[name] = config
            
            # Update the config_manager with the new configuration
            config_manager.size_config = updated_config
            config_manager.save_size_config()
            
            # Create a new dataframe with the updated configuration
            new_df = pd.DataFrame(
                [[name] + list(config.values()) for name, config in updated_config.items()],
                columns=['Name'] + (list(next(iter(updated_config.values())).keys()) if updated_config else [])
            )
            
            return new_df, t('size_config_updated', config_manager.language)

        def add_color_config(df):
            """Add a new empty row to the color configuration table."""
            new_row = pd.DataFrame([['' for _ in df.columns]], columns=df.columns)
            updated_df = pd.concat([df, new_row], ignore_index=True)
            return updated_df, t('color_config_row_added', config_manager.language)

        def update_color_config(df):
            """Save all changes made to the color configuration table and remove empty rows."""
            updated_config = {}
            for _, row in df.iterrows():
                name = row['Name']
                if name and not row.iloc[1:].isna().all():  # Check if name exists and not all other fields are empty
                    config = row.to_dict()
                    config.pop('Name')
                    updated_config[name] = config
            
            # Update the config_manager with the new configuration
            config_manager.color_config = updated_config
            config_manager.save_color_config()
            
            # Create a new dataframe with the updated configuration
            new_df = pd.DataFrame(
                [[name] + list(config.values()) for name, config in updated_config.items()],
                columns=['Name', 'R', 'G', 'B', 'Notes']
            )
            
            return new_df, t('color_config_updated', config_manager.language)

        lang_dropdown.change(
            update_language,
            inputs=[lang_dropdown],
            outputs=[title, input_image, lang_dropdown, photo_type, photo_sheet_size, preset_color, background_color, 
                    sheet_rows, sheet_cols, layout_only, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                    compress, change_background, rotate, resize, add_crop_lines, process_btn, output_image, size_df, color_df,
                    corrected_output, notification, key_param_tab, advanced_settings_tab, config_management_tab, confirm_advanced_settings,
                    size_config_tab, color_config_tab, result_tab, corrected_image_tab,
                    size_df, color_df, add_size_btn, update_size_btn,
                    add_color_btn, update_color_btn, config_notification]
        )

        confirm_advanced_settings.click(
            confirm_advanced_settings_fn,
            inputs=[yolov8_path, yunet_path, rmbg_path, size_config, color_config],
            outputs=[size_df, color_df, photo_type, photo_sheet_size, preset_color]
        )

        preset_color.change(
            update_background_color,
            inputs=[preset_color, lang_dropdown],
            outputs=[background_color]
        )

        background_color.change(
            update_preset_color,
            inputs=[background_color, lang_dropdown],
            outputs=[preset_color]
        )

        add_size_btn.click(add_size_config, inputs=[size_df], outputs=[size_df, config_notification])
        update_size_btn.click(update_size_config, inputs=[size_df], outputs=[size_df, config_notification])

        add_color_btn.click(add_color_config, inputs=[color_df], outputs=[color_df, config_notification])
        update_color_btn.click(update_color_config, inputs=[color_df], outputs=[color_df, config_notification])

        process_btn.click(
            process_and_display,
            inputs=[input_image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                    photo_type, photo_sheet_size, background_color, compress, change_background, 
                    rotate, resize, sheet_rows, sheet_cols, layout_only, add_crop_lines],
            outputs=[output_image, corrected_output]
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiYing Photo Processing System")
    parser.add_argument("--lang", type=str, choices=['en', 'zh'], default=get_language(), help="Specify the language (en/zh)")
    args = parser.parse_args()

    initial_language = args.lang
    demo = create_demo(initial_language)
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

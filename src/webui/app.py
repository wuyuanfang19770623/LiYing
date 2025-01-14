import gradio as gr
import os
import sys
import re
import cv2
import locale
import json
import argparse
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

PROJECT_ROOT = src_dir
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TOOL_DIR = os.path.join(PROJECT_ROOT, 'tool')
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
    rgb_match = re.match(r'rgb[a]?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)', color_string)
    if rgb_match:
        return list(map(int, rgb_match.groups()))
    return [255, 255, 255]

def process_image(img_path, yolov8_path, yunet_path, rmbg_path, photo_requirements, photo_type, photo_sheet_size, rgb_list, compress=False, change_background=False, rotate=False, resize=True, sheet_rows=3, sheet_cols=3):
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
    photo_sheet_cv = generator.generate_photo_sheet(processor.photo.image, sheet_rows, sheet_cols, rotate)

    return {
        'final_image': photo_sheet_cv,
        'corrected_image': processor.photo.image,
    }

def process_and_display(image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, photo_type, photo_sheet_size, background_color, compress, change_background, rotate, resize, sheet_rows, sheet_cols):
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
        size_config,
        color_config,
        photo_type=photo_type,
        photo_sheet_size=photo_sheet_size,
        rgb_list=rgb_list,
        compress=compress,
        change_background=change_background,
        rotate=rotate,
        resize=resize,
        sheet_rows=sheet_rows,
        sheet_cols=sheet_cols
    )
    
    os.remove(temp_image_path)
    final_image_rgb = cv2.cvtColor(result['final_image'], cv2.COLOR_BGR2RGB)
    corrected_image_rgb = cv2.cvtColor(result['corrected_image'], cv2.COLOR_BGR2RGB)
    
    return final_image_rgb, corrected_image_rgb

def create_demo(initial_language):
    """Create the Gradio demo interface."""
    config_manager = ConfigManager(language=initial_language)
    config_manager.load_configs()
    photo_requirements = PhotoRequirements(language=initial_language)

    photo_size_configs = config_manager.get_photo_size_configs()
    sheet_size_configs = config_manager.get_sheet_size_configs()

    photo_size_choices = list(photo_size_configs.keys())
    sheet_size_choices = list(sheet_size_configs.keys())

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        language = gr.State(initial_language)

        title = gr.Markdown(f"# {t('title', initial_language)}")
        
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
                        background_color = gr.ColorPicker(label=t('background_color', initial_language), value="#FFFFFF")
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
            
            new_photo_size_configs = config_manager.get_photo_size_configs()
            new_sheet_size_configs = config_manager.get_sheet_size_configs()
            
            new_photo_size_choices = list(new_photo_size_configs.keys())
            new_sheet_size_choices = list(new_sheet_size_configs.keys())

            return {
                title: gr.update(value=f"# {t('title', lang)}"),
                input_image: gr.update(label=t('upload_photo', lang)),
                lang_dropdown: gr.update(label=t('language', lang)),
                photo_type: gr.update(choices=new_photo_size_choices, label=t('photo_type', lang), value=new_photo_size_choices[0] if new_photo_size_choices else None),
                photo_sheet_size: gr.update(choices=new_sheet_size_choices, label=t('photo_sheet_size', lang), value=new_sheet_size_choices[0] if new_sheet_size_choices else None),
                background_color: gr.update(label=t('background_color', lang)),
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
                process_btn: gr.update(value=t('process_btn', lang)),
                output_image: gr.update(label=t('final_image', lang)),
                corrected_output: gr.update(label=t('corrected_image', lang)),
                notification: gr.update(label=t('notification', lang)),
                key_param_tab: gr.update(label=t('key_param', lang)),
                advanced_settings_tab: gr.update(label=t('advanced_settings', lang)),
                result_tab: gr.update(label=t('result', lang)),
                corrected_image_tab: gr.update(label=t('corrected_image', lang)),
            }

        def process_and_display(image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, photo_type, photo_sheet_size, background_color, compress, change_background, rotate, resize, sheet_rows, sheet_cols):
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
                change_background=change_background,
                rotate=rotate,
                resize=resize,
                sheet_rows=sheet_rows,
                sheet_cols=sheet_cols
            )
            
            os.remove(temp_image_path)
            final_image_rgb = cv2.cvtColor(result['final_image'], cv2.COLOR_BGR2RGB)
            corrected_image_rgb = cv2.cvtColor(result['corrected_image'], cv2.COLOR_BGR2RGB)
            
            return final_image_rgb, corrected_image_rgb

        lang_dropdown.change(
            update_language,
            inputs=[lang_dropdown],
            outputs=[title, input_image, lang_dropdown, photo_type, photo_sheet_size, background_color, 
                     sheet_rows, sheet_cols, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                     compress, change_background, rotate, resize, process_btn, output_image, 
                     corrected_output, notification, key_param_tab, advanced_settings_tab,
                     result_tab, corrected_image_tab]
        )

        process_btn.click(
            process_and_display,
            inputs=[input_image, yolov8_path, yunet_path, rmbg_path, size_config, color_config, 
                    photo_type, photo_sheet_size, background_color, compress, change_background, 
                    rotate, resize, sheet_rows, sheet_cols],
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

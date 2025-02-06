import locale

import click
import os
import sys

# Set the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(current_dir, 'model')
TOOL_DIR = os.path.join(current_dir, 'tool')

# Add data and model directories to sys.path
sys.path.extend([DATA_DIR, MODEL_DIR, TOOL_DIR])

# Set src directory as PYTHONPATH
sys.path.insert(0, current_dir)

from tool.ImageProcessor import ImageProcessor
from tool.PhotoSheetGenerator import PhotoSheetGenerator

from tool.PhotoRequirements import PhotoRequirements


class RGBListType(click.ParamType):
    name = 'rgb_list'

    def convert(self, value, param, ctx):
        if value:
            try:
                return tuple(int(x) for x in value.split(','))
            except ValueError:
                self.fail(f'{value} is not a valid RGB list format. Expected format: INTEGER,INTEGER,INTEGER.')
        return 0, 0, 0  # Default value


def get_language():
    # Get custom language environment variable
    language = os.getenv('CLI_LANGUAGE', '')
    if language == '':
        # Get system language
        system_language, _ = locale.getdefaultlocale()
        language = 'en' if system_language and system_language.startswith('en') else 'zh'
        return language
    return language


# Define multilingual support messages
messages = {
    'en': {
        'corrected_saved': 'Corrected image saved to {path}',
        'background_saved': 'Background-changed image saved to {path}',
        'resized_saved': 'Resized image saved to {path}',
        'sheet_saved': 'Photo sheet saved to {path}',
    },
    'zh': {
        'corrected_saved': '裁剪并修正后的图像已保存到 {path}',
        'background_saved': '替换背景后的图像已保存到 {path}',
        'resized_saved': '调整尺寸后的图像已保存到 {path}',
        'sheet_saved': '照片表格已保存到 {path}',
    }
}


def echo_message(key, **kwargs):
    lang = get_language()
    message = messages.get(lang, messages['en']).get(key, '')
    click.echo(message.format(**kwargs))


@click.command()
@click.argument('img_path', type=click.Path(exists=True, resolve_path=True))
@click.option('-y', '--yolov8-model-path', type=click.Path(),
              default=os.path.join(MODEL_DIR, 'yolov8n-pose.onnx'),
              help='Path to YOLOv8 model' if get_language() == 'en' else 'YOLOv8 模型路径')
@click.option('-u', '--yunet-model-path', type=click.Path(),
              default=os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx'),
              help='Path to YuNet model' if get_language() == 'en' else 'YuNet 模型路径')
@click.option('-r', '--rmbg-model-path', type=click.Path(),
              default=os.path.join(MODEL_DIR, 'RMBG-1.4-model.onnx'),
              help='Path to RMBG model' if get_language() == 'en' else 'RMBG 模型路径')
@click.option('-sz', '--size-config', type=click.Path(exists=True),
              default=os.path.join(DATA_DIR, f'size_{get_language()}.csv'),
              help='Path to size configuration file' if get_language() == 'en' else '尺寸配置文件路径')
@click.option('-cl', '--color-config', type=click.Path(exists=True),
              default=os.path.join(DATA_DIR, f'color_{get_language()}.csv'),
              help='Path to color configuration file' if get_language() == 'en' else '颜色配置文件路径')
@click.option('-b', '--rgb-list', type=RGBListType(), default='0,0,0',
              help='RGB channel values list (comma-separated) for image composition' if get_language() == 'en' else 'RGB 通道值列表（英文逗号分隔），用于图像合成')
@click.option('-s', '--save-path', type=click.Path(), default='output.jpg',
              help='Path to save the output image' if get_language() == 'en' else '保存路径')
@click.option('-p', '--photo-type', type=str, default='One Inch' if get_language() == 'en' else '一寸',
              help='Photo types' if get_language() == 'en' else '照片类型')
@click.option('-ps', '--photo-sheet-size', type=str, default='Five Inch' if get_language() == 'en' else '五寸',
              help='Size of the photo sheet' if get_language() == 'en' else '选择照片表格的尺寸')
@click.option('-c', '--compress/--no-compress', default=False,
              help='Whether to compress the image' if get_language() == 'en' else '是否压缩图像（使用 AGPicCompress 压缩）')
@click.option('-sv', '--save-corrected/--no-save-corrected', default=False,
              help='Whether to save the corrected image' if get_language() == 'en' else '是否保存修正图像后的图片')
@click.option('-bg', '--change-background/--no-change-background', default=False,
              help='Whether to change the background' if get_language() == 'en' else '是否替换背景')
@click.option('-sb', '--save-background/--no-save-background', default=False,
              help='Whether to save the image with changed background' if get_language() == 'en' else '是否保存替换背景后的图像')
@click.option('-lo', '--layout-only', is_flag=True, default=False,
              help='Only layout the photo without changing background' if get_language() == 'en' else '仅排版照片，不更换背景')
@click.option('-sr', '--sheet-rows', type=int, default=3,
              help='Number of rows in the photo sheet' if get_language() == 'en' else '照片表格的行数')
@click.option('-sc', '--sheet-cols', type=int, default=3,
              help='Number of columns in the photo sheet' if get_language() == 'en' else '照片表格的列数')
@click.option('-rt', '--rotate/--no-rotate', default=False,
              help='Whether to rotate the photo by 90 degrees' if get_language() == 'en' else '是否旋转照片90度')
@click.option('-rs', '--resize/--no-resize', default=True,
              help='Whether to resize the image' if get_language() == 'en' else '是否调整图像尺寸')
@click.option('-sz', '--save-resized/--no-save-resized', default=False,
              help='Whether to save the resized image' if get_language() == 'en' else '是否保存调整尺寸后的图像')
@click.option('-al', '--add-crop-lines/--no-add-crop-lines', default=True, 
              help='Add crop lines to the photo sheet' if get_language() == 'en' else '在照片表格上添加裁剪线')
def cli(img_path, yolov8_model_path, yunet_model_path, rmbg_model_path, size_config, color_config, rgb_list, save_path, 
        photo_type, photo_sheet_size, compress, save_corrected, change_background, save_background, layout_only, sheet_rows, 
        sheet_cols, rotate, resize, save_resized, add_crop_lines):
    # Create an instance of the image processor
    processor = ImageProcessor(img_path, yolov8_model_path, yunet_model_path, rmbg_model_path, rgb_list, y_b=compress)
    photo_requirements = PhotoRequirements(get_language(), size_config, color_config)
    # Crop and correct image
    processor.crop_and_correct_image()
    if save_corrected:
        corrected_path = os.path.splitext(save_path)[0] + '_corrected' + os.path.splitext(save_path)[1]
        processor.save_photos(corrected_path, compress)
        echo_message('corrected_saved', path=corrected_path)

    # Optional background change
    if change_background and not layout_only:
        processor.change_background()
        if save_background:
            background_path = os.path.splitext(save_path)[0] + '_background' + os.path.splitext(save_path)[1]
            processor.save_photos(background_path, compress)
            echo_message('background_saved', path=background_path)

    # Optional resizing
    if resize:
        processor.resize_image(photo_type)
        if save_resized:
            resized_path = os.path.splitext(save_path)[0] + '_resized' + os.path.splitext(save_path)[1]
            processor.save_photos(resized_path, compress)
            echo_message('resized_saved', path=resized_path)

    # Generate photo sheet
    # Set photo sheet size
    sheet_info = photo_requirements.get_resize_image_list(photo_sheet_size)
    sheet_width, sheet_height, sheet_resolution = sheet_info['width'], sheet_info['height'], sheet_info['resolution']
    generator = PhotoSheetGenerator((sheet_width, sheet_height), sheet_resolution)
    photo_sheet_cv = generator.generate_photo_sheet(processor.photo.image, sheet_rows, sheet_cols, rotate, add_crop_lines)
    sheet_path = os.path.splitext(save_path)[0] + '_sheet' + os.path.splitext(save_path)[1]
    generator.save_photo_sheet(photo_sheet_cv, sheet_path)
    echo_message('sheet_saved', path=sheet_path)


if __name__ == "__main__":
    cli()

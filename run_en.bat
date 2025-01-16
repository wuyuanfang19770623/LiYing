@echo off
set CLI_LANGUAGE=en
setlocal enabledelayedexpansion

REM Get the current batch file directory
set SCRIPT_DIR=%~dp0

REM Set Python interpreter path and project directory
set PYTHON_EXE=%SCRIPT_DIR%_myPython\python.exe
set SCRIPT_PATH=%SCRIPT_DIR%src\main.py

REM Check if files or directories were dragged and dropped
if "%~1"=="" (
    echo Please drag and drop image files or directories onto this script
    pause
    exit /b
)

REM Get the dropped path
set INPUT_PATH=%~1

echo                  LiYing
echo Github: https://github.com/aoguai/LiYing
echo LICENSE AGPL-3.0 license
echo ----------------------------------------

REM Prompt user for input parameters
set /p "layout_only=Layout only without changing background (yes/no, default is no): "
if /i "!layout_only!"=="yes" (
    set layout_only=--layout-only
    set change_background=--no-change-background
    set save_background=--no-save-background
    set rgb_list=255,255,255
) else (
    set layout_only=
    set /p "change_background=Change background (yes/no, default is no): "
    if /i "!change_background!"=="yes" (
        set change_background=--change-background
        set /p "rgb_list=Enter RGB channel values (comma separated, default is 255,255,255): "
        if "!rgb_list!"=="red" set rgb_list=255,0,0
        if "!rgb_list!"=="blue" set rgb_list=12,92,165
        if "!rgb_list!"=="white" set rgb_list=255,255,255
        if "!rgb_list!"=="" set rgb_list=255,255,255
        set /p "save_background=Save images with changed background (yes/no, default is no): "
        if /i "!save_background!"=="yes" (
            set save_background=--save-background
        ) else (
            set save_background=--no-save-background
        )
    ) else (
        set change_background=--no-change-background
        set save_background=--no-save-background
        set rgb_list=255,255,255
    )
)

set /p "resize=Resize images (yes/no, default is yes): "
if /i "!resize!"=="no" (
    set resize=--no-resize
    set save_resized=--no-save-resized
) else (
    set resize=--resize
    set /p "save_resized=Save resized images (yes/no, default is no): "
    if /i "!save_resized!"=="yes" (
        set save_resized=--save-resized
    ) else (
        set save_resized=--no-save-resized
    )
    set /p "photo_type=Enter photo type (default is one_inch): "
    if "!photo_type!"=="" set photo_type=one_inch
)

set /p "photo_sheet_size=Enter photo sheet size (default is five_inch): "
if "!photo_sheet_size!"=="" set photo_sheet_size=five_inch

set /p "compress=Compress images (yes/no, default is no): "
if /i "!compress!"=="yes" (
    set compress=--compress
) else (
    set compress=--no-compress
)

set /p "save_corrected=Save corrected images (yes/no, default is no): "
if /i "!save_corrected!"=="yes" (
    set save_corrected=--save-corrected
) else (
    set save_corrected=--no-save-corrected
)

set /p "sheet_rows=Enter the number of rows in the photo sheet (default is 3): "
if "!sheet_rows!"=="" set sheet_rows=3

set /p "sheet_cols=Enter the number of columns in the photo sheet (default is 3): "
if "!sheet_cols!"=="" set sheet_cols=3

set /p "rotate=Rotate photos 90 degrees (yes/no, default is no): "
if /i "!rotate!"=="yes" (
    set rotate=--rotate
) else (
    set rotate=--no-rotate
)

set /p "add_crop_lines=Add crop lines to the photo sheet (yes/no, default is yes): "
if /i "!add_crop_lines!"=="no" (
    set add_crop_lines=--no-add-crop-lines
) else (
    set add_crop_lines=--add-crop-lines
)

REM Check if the dropped item is a file or a directory
if exist "%INPUT_PATH%\" (
    REM If it's a directory, iterate through all jpg and png files in it
    for %%f in ("%INPUT_PATH%\*.jpg" "%INPUT_PATH%\*.png") do (
        REM Extract folder path and file name
        set "INPUT_FILE=%%~ff"
        set "OUTPUT_PATH=%%~dpnf_output%%~xf"
        
        REM Execute Python script to process the image
        start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "%%~ff" -b !rgb_list! -s "%%~dpnf_output%%~xf" -p !photo_type! --photo-sheet-size !photo_sheet_size! !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! !layout_only! !add_crop_lines! & pause"
    )
) else (
    REM If it's a file, process the file directly
    set INPUT_DIR=%~dp1
    set INPUT_FILE=%~nx1
    set OUTPUT_PATH=%INPUT_DIR%%~n1_output%~x1
    
    REM Due to setlocal enabledelayedexpansion, use !variable_name! to reference variables
    start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "!INPUT_PATH!" -b !rgb_list! -s "!OUTPUT_PATH!" -p !photo_type! --photo-sheet-size !photo_sheet_size! !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! !layout_only! !add_crop_lines! & pause"
)

pause

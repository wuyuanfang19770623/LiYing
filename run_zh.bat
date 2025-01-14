@echo off
set CLI_LANGUAGE=zh
setlocal enabledelayedexpansion

REM 获取当前批处理文件目录
set SCRIPT_DIR=%~dp0

REM 设置Python解释器路径和项目目录
set PYTHON_EXE=%SCRIPT_DIR%_myPython\python.exe
set SCRIPT_PATH=%SCRIPT_DIR%src\main.py

REM 检查是否有文件或目录被拖放
if "%~1"=="" (
    echo 请将图像文件或目录拖放到此脚本上
    pause
    exit /b
)

REM 获取拖放的路径
set INPUT_PATH=%~1

echo                  LiYing
echo Github: https://github.com/aoguai/LiYing
echo LICENSE AGPL-3.0 license
echo ----------------------------------------

REM 提示用户输入参数
set /p "resize=是否调整图像尺寸（是/否，默认是）："
if /i "!resize!"=="否" (
    set resize=--no-resize
    set save_resized=--no-save-resized
) else (
    set resize=--resize
    set /p "save_resized=是否保存调整尺寸后的图像（是/否，默认否）："
    if /i "!save_resized!"=="是" (
        set save_resized=--save-resized
    ) else (
        set save_resized=--no-save-resized
    )
    set /p "photo_type=输入照片类型（默认为一寸）："
    if "!photo_type!"=="" set photo_type=一寸
)

set /p "photo_sheet_size=输入照片表格尺寸（默认为五寸）："
if "!photo_sheet_size!"=="" set photo_sheet_size=五寸

set /p "compress=是否压缩图像（是/否，默认否）："
if /i "!compress!"=="是" (
    set compress=--compress
) else (
    set compress=--no-compress
)

set /p "save_corrected=是否保存修正后的图像（是/否，默认否）："
if /i "!save_corrected!"=="是" (
    set save_corrected=--save-corrected
) else (
    set save_corrected=--no-save-corrected
)

set /p "change_background=是否更换背景（是/否，默认否）："
if /i "!change_background!"=="是" (
    set change_background=--change-background
    set /p "rgb_list=输入RGB通道值（逗号分隔，默认为255,255,255）："
    if "!rgb_list!"=="红色" set rgb_list=255,0,0
    if "!rgb_list!"=="蓝色" set rgb_list=12,92,165
    if "!rgb_list!"=="白色" set rgb_list=255,255,255
    if "!rgb_list!"=="" set rgb_list=255,255,255
    set /p "save_background=是否保存更换背景后的图像（是/否，默认否）："
    if /i "!save_background!"=="是" (
        set save_background=--save-background
    ) else (
        set save_background=--no-save-background
    )
) else (
    set change_background=--no-change-background
    set save_background=--no-save-background
    set rgb_list=255,255,255
)

set /p "sheet_rows=输入照片表格的行数（默认为3）："
if "!sheet_rows!"=="" set sheet_rows=3

set /p "sheet_cols=输入照片表格的列数（默认为3）："
if "!sheet_cols!"=="" set sheet_cols=3

set /p "rotate=是否旋转照片90度（是/否，默认否）："
if /i "!rotate!"=="是" (
    set rotate=--rotate
) else (
    set rotate=--no-rotate
)

REM 检查拖放的项目是文件还是目录
if exist "%INPUT_PATH%\" (
    REM 如果是目录，遍历其中所有的jpg和png文件
    for %%f in ("%INPUT_PATH%\*.jpg" "%INPUT_PATH%\*.png") do (
        REM 提取文件夹路径和文件名
        set "INPUT_FILE=%%~ff"
        set "OUTPUT_PATH=%%~dpnf_output%%~xf"
        
        REM 执行Python脚本来处理图像
        start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "%%~ff" -b !rgb_list! -s "%%~dpnf_output%%~xf" -p !photo_type! --photo-sheet-size !photo_sheet_size! !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! & pause"
    )
) else (
    REM 如果是文件，直接处理该文件
    set INPUT_DIR=%~dp1
    set INPUT_FILE=%~nx1
    set OUTPUT_PATH=%INPUT_DIR%%~n1_output%~x1
    
    REM 由于使用了setlocal enabledelayedexpansion，使用!variable_name!来引用变量
    start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "!INPUT_PATH!" -b !rgb_list! -s "!OUTPUT_PATH!" -p !photo_type! --photo-sheet-size !photo_sheet_size! !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! & pause"
)

pause

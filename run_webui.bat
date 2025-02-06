@echo off
setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%python-embed\python.exe
cd src\webui
%PYTHON_EXE% app.py
pause

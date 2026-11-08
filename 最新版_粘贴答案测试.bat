@echo off
chcp 65001 >nul
setlocal
set "RESIDUAL_INPUT="
if not "%~1"=="" set "RESIDUAL_INPUT=%~f1"
cd /d "%~dp0versions\20261020-residual-gate\lite-questionnaire-model"
if errorlevel 1 goto folder_error
if defined RESIDUAL_INPUT goto input_file
python -B residual_demo.py
goto finished
:input_file
python -B residual_demo.py "%RESIDUAL_INPUT%"
:finished
if errorlevel 1 pause
endlocal
exit /b
:folder_error
echo 找不到残差门控实验目录，请将此脚本放在本包根目录。
pause
exit /b 1

@echo off
REM One-click setup + launch for Windows. Double-click this file.
cd /d "%~dp0"
if not exist venv ( python -m venv venv )
call venv\Scripts\activate.bat
pip install -r requirements.txt
if not exist pharmacy.db ( python train.py )
streamlit run app.py

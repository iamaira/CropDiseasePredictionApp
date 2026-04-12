@echo off
REM Run this from the project root in PowerShell or Command Prompt.
REM Make sure Python 3.11+ is installed and available on PATH.
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-render.txt
python app.py
pause

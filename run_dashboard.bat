@echo off
REM  Launch the Composite Similarity Dashboard
REM  PyTorch is installed to C:\pt to avoid Windows MAX_PATH issues.
set PYTHONPATH=C:\pt;%PYTHONPATH%
python -m streamlit run "%~dp0app.py" --server.port 8502

@echo off
chcp 65001

call venv\python.exe webui_dataset.py

@echo 请按任意键继续
call pause
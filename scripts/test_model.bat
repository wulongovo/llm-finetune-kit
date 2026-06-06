@echo off
echo 测试微调模型...
call venv\Scripts\activate.bat
python inference.py --adapter outputs/final_adapter --interactive
pause

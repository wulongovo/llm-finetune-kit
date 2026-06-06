@echo off
echo ========================================
echo  LLM Fine-tune Kit - Qwen3.5-Heretic
echo ========================================
echo.

call venv\Scripts\activate.bat

echo [1/2] 生成示例数据（如果不存在）...
python -c "from src.data_processor import create_sample_data; create_sample_data()"

echo.
echo [2/2] 开始 LoRA 微调...
python train.py --config configs/qwen35_heretic.yaml

echo.
echo ✅ 训练完成！
echo 测试: python inference.py --adapter outputs/qwen35-heretic-lora/final_adapter --interactive
pause

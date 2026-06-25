@echo off
REM DeepSpeed 多卡训练 (Windows)
set CONFIG=%1
set DEEPSPEED=%2
if "%CONFIG%"=="" set CONFIG=configs\lora_config.yaml
if "%DEEPSPEED%"=="" set DEEPSPEED=configs\deepspeed_zero2.json

echo ==========================================
echo   DeepSpeed 多卡训练
echo   训练配置: %CONFIG%
echo   DeepSpeed: %DEEPSPEED%
echo ==========================================

deepspeed --num_gpus=2 train.py --config %CONFIG% --deepspeed %DEEPSPEED%

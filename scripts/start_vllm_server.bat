@echo off
REM 启动 vLLM 推理服务 (Windows)

set MODEL=%1
set PORT=%2
if "%MODEL%"=="" set MODEL=outputs\merged_model
if "%PORT%"=="" set PORT=8000

echo ==========================================
echo   vLLM 推理服务
echo   模型: %MODEL%
echo   端口: %PORT%
echo ==========================================

python -m vllm.entrypoints.openai.api_server --model %MODEL% --port %PORT% --dtype bfloat16

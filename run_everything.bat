@echo off
echo ========================================
echo Bone Cancer Prediction - Full Pipeline
echo ========================================

REM Set Python path
set PYTHONPATH=T:\bone_can_pre

echo.
echo [1/3] Training the model...
echo This may take 10-30 minutes depending on your hardware
echo.

python scripts\train.py --epochs 10 --batch-size 16

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Training completed successfully!
echo Model saved to: models\efficientnet_b0_best.pt
echo.

echo [3/3] Starting the web application...
echo The app will be available at: http://localhost:8000
echo.

set BONE_CKPT=models\efficientnet_b0_best.pt
uvicorn app.server:app --host 0.0.0.0 --port 8000

pause

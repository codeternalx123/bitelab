@echo off
REM Quick Start Script for AI Nutrition System
REM This script guides you through setup and first run

echo.
echo ============================================================
echo    AI NUTRITION SYSTEM - QUICK START
echo    Database Population + Computer Vision Setup
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check current directory
if not exist "flaskbackend" (
    echo [ERROR] Please run this script from the wellomex root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

cd flaskbackend
echo [OK] Changed to flaskbackend directory
echo.

REM Step 1: Check .env file
echo ============================================================
echo STEP 1: API Key Setup
echo ============================================================
echo.

if exist ".env" (
    echo [OK] .env file exists
    findstr /C:"USDA_API_KEY" .env >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] USDA_API_KEY not found in .env
        echo.
        echo To get your FREE API key:
        echo 1. Visit: https://fdc.nal.usda.gov/api-key-signup.html
        echo 2. Fill the form (takes 2 minutes^)
        echo 3. Check your email for the API key
        echo 4. Add this line to flaskbackend\.env:
        echo    USDA_API_KEY=your_key_here
        echo.
        echo NOTE: You can still continue with Open Food Facts only (no key needed^)
        echo       This will give you 3,000-5,000 foods instead of 10,000+
        echo.
    ) else (
        echo [OK] USDA_API_KEY found in .env
    )
) else (
    echo [INFO] Creating .env file...
    echo # AI Nutrition System Configuration > .env
    echo. >> .env
    echo # Get your FREE USDA API key from: >> .env
    echo # https://fdc.nal.usda.gov/api-key-signup.html >> .env
    echo USDA_API_KEY= >> .env
    echo.
    echo [OK] .env file created
    echo.
    echo Please add your USDA API key to flaskbackend\.env
    echo Open the file and replace: USDA_API_KEY= 
    echo                      with: USDA_API_KEY=your_key_here
    echo.
)

pause
echo.

REM Step 2: Install dependencies
echo ============================================================
echo STEP 2: Install Dependencies
echo ============================================================
echo.
echo This will install:
echo - API integration: requests, python-dotenv, urllib3
echo - Computer vision: torch, torchvision, ultralytics, opencv
echo - Data processing: albumentations, pillow, numpy
echo - Model export: onnx, onnxruntime
echo - Visualization: matplotlib, seaborn
echo.
echo Estimated download size: ~2-3 GB
echo Estimated time: 5-10 minutes (depending on internet speed^)
echo.

set /p install="Install dependencies now? (y/n): "
if /i not "%install%"=="y" (
    echo Skipped. You can install later with: pip install -r requirements.txt
    goto :skip_install
)

echo.
echo [INFO] Installing dependencies...
echo.

python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed. Please check error messages above.
    pause
    exit /b 1
)

echo.
echo [OK] All dependencies installed successfully
echo.

:skip_install
pause
echo.

REM Step 3: Populate database
echo ============================================================
echo STEP 3: Populate Food Database
echo ============================================================
echo.
echo This will download 10,000+ foods from:
echo - USDA FoodData Central (350,000+ foods available^)
echo - Open Food Facts (2,000,000+ products available^)
echo - Regional databases (7 regions^)
echo.
echo Estimated time: 1-2 hours (automated^)
echo Cost: FREE (using free API tiers^)
echo.

set /p populate="Populate database now? (y/n): "
if /i not "%populate%"=="y" (
    echo Skipped. You can populate later with: python populate_food_database.py
    goto :skip_populate
)

echo.
echo [INFO] Starting database population...
echo       This will take 1-2 hours. You can stop anytime with Ctrl+C
echo.

python populate_food_database.py

if errorlevel 1 (
    echo.
    echo [WARNING] Population completed with warnings.
    echo          Database may have fewer than 10,000 foods if USDA key not configured.
    echo.
) else (
    echo.
    echo [OK] Database population complete!
    echo.
)

:skip_populate
pause
echo.

REM Step 4: Download training data
echo ============================================================
echo STEP 4: Download Food-101 Training Dataset
echo ============================================================
echo.
echo This will download 101,000 food images for AI training
echo - Size: 4.65 GB
echo - Categories: 101 food types
echo - Estimated time: 30-60 minutes
echo.
echo NOTE: Only needed if you plan to train the computer vision model
echo       You can skip this if you only want to use the food database
echo.

set /p download="Download Food-101 dataset now? (y/n): "
if /i not "%download%"=="y" (
    echo Skipped. You can download later with: python download_food101.py
    goto :skip_download
)

echo.
echo [INFO] Starting Food-101 download...
echo       This will take 30-60 minutes. You can stop anytime with Ctrl+C
echo.

python download_food101.py

if errorlevel 1 (
    echo.
    echo [ERROR] Download failed. Please check error messages above.
    echo         You can retry later with: python download_food101.py
    echo.
) else (
    echo.
    echo [OK] Food-101 dataset downloaded successfully!
    echo.
)

:skip_download
pause
echo.

REM Step 5: Summary and next steps
echo ============================================================
echo SETUP COMPLETE! 
echo ============================================================
echo.
echo What you have now:
echo - [%] Food database with API integration
echo - [%] Computer vision preprocessing pipeline
echo - [%] YOLOv8 training infrastructure
echo - [%] Food-101 dataset (if downloaded^)
echo.
echo Next steps:
echo.
echo 1. TRAIN AI MODEL (requires GPU^):
echo    python -c "from app.ai_nutrition.cv.cv_yolov8_model import *; detector = test_yolov8_setup()"
echo.
echo 2. TEST DATABASE:
echo    python -m app.ai_nutrition.database.enhanced_food_database stats
echo    python -m app.ai_nutrition.database.enhanced_food_database search "chicken"
echo.
echo 3. TEST PREPROCESSING:
echo    python -m app.ai_nutrition.cv.cv_image_preprocessing
echo.
echo 4. READ DOCUMENTATION:
echo    - SETUP_AND_PHASE2_GUIDE.md (complete guide^)
echo    - SESSION_SUMMARY.md (what we built today^)
echo    - FOOD_DATABASE_COMPLETE.md (database details^)
echo.
echo 5. START API SERVER:
echo    uvicorn app.main:app --reload --port 8000
echo.
echo ============================================================
echo Need help? Check the documentation files listed above!
echo ============================================================
echo.

pause

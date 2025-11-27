@echo off
REM Quick Setup Script for FatSecret API Integration
REM Windows Batch File

echo ================================================================================
echo AI NUTRITION SCANNER - FATSECRET API SETUP
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Install required packages
echo [STEP 1/5] Installing required Python packages...
pip install requests numpy pandas scikit-learn python-dotenv

echo.
echo [OK] Packages installed
echo.

REM Check for credentials
echo [STEP 2/5] Checking for FatSecret API credentials...
if "%FATSECRET_CLIENT_ID%"=="" (
    echo.
    echo [WARNING] FATSECRET_CLIENT_ID environment variable not set
    echo.
    echo Please set your FatSecret API credentials:
    echo.
    echo 1. Sign up at: https://platform.fatsecret.com/api/
    echo 2. Create an application
    echo 3. Copy your Client ID and Client Secret
    echo.
    set /p CLIENT_ID="Enter your Client ID: "
    set /p CLIENT_SECRET="Enter your Client Secret: "
    
    REM Create .env file
    echo FATSECRET_CLIENT_ID=%CLIENT_ID%> .env
    echo FATSECRET_CLIENT_SECRET=%CLIENT_SECRET%>> .env
    
    echo.
    echo [OK] Credentials saved to .env file
    echo.
    
    REM Set for current session
    set FATSECRET_CLIENT_ID=%CLIENT_ID%
    set FATSECRET_CLIENT_SECRET=%CLIENT_SECRET%
) else (
    echo [OK] Credentials found in environment
)

echo.

REM Test API connection
echo [STEP 3/5] Testing FatSecret API connection...
python fatsecret_client.py
if errorlevel 1 (
    echo.
    echo [ERROR] API connection test failed
    echo Please check your credentials and internet connection
    pause
    exit /b 1
)

echo.
echo [OK] API connection successful
echo.

REM Train models
echo [STEP 4/5] Training ML models...
echo This may take 5-10 minutes...
echo.
python api_model_trainer.py
if errorlevel 1 (
    echo.
    echo [ERROR] Model training failed
    pause
    exit /b 1
)

echo.
echo [OK] Models trained successfully
echo.

REM Test scanner
echo [STEP 5/5] Testing food scanner...
python api_food_scanner.py
if errorlevel 1 (
    echo.
    echo [ERROR] Scanner test failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] SETUP COMPLETE!
echo ================================================================================
echo.
echo You can now use the AI Food Scanner with FatSecret API
echo.
echo Quick test:
echo   python api_food_scanner.py
echo.
echo Next steps:
echo   1. Read API_INTEGRATION_README.md for detailed documentation
echo   2. Integrate api_food_scanner.py into your application
echo   3. Replace old food_composition_database.py calls
echo.
echo Models are saved in: models/
echo Training data is saved in: training_data.csv
echo.
echo ================================================================================
pause

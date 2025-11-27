"""
Quick Setup Script for FatSecret API Integration
Automates the setup process
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def check_python():
    """Check Python version"""
    print("[STEP 1/6] Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. You have {}.{}.{}".format(
            version.major, version.minor, version.micro
        ))
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} found")
    return True


def install_packages():
    """Install required Python packages"""
    print("\n[STEP 2/6] Installing required packages...")
    
    packages = [
        'requests',
        'numpy',
        'pandas',
        'scikit-learn',
        'python-dotenv'
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Warning: Could not install {package}")
    
    print("✅ Packages installed")
    return True


def setup_credentials():
    """Set up FatSecret API credentials"""
    print("\n[STEP 3/6] Setting up FatSecret API credentials...")
    
    # Check if already set
    client_id = os.getenv('FATSECRET_CLIENT_ID')
    client_secret = os.getenv('FATSECRET_CLIENT_SECRET')
    
    if client_id and client_secret:
        print("✅ Credentials found in environment")
        return True
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ Credentials found in .env file")
        # Load .env
        from dotenv import load_dotenv
        load_dotenv()
        return True
    
    # Need to set up
    print("\n⚠️  FatSecret API credentials not found")
    print("\nTo get your credentials:")
    print("1. Go to: https://platform.fatsecret.com/api/")
    print("2. Sign up for a free account")
    print("3. Create an application")
    print("4. Copy your Client ID and Client Secret")
    print()
    
    choice = input("Do you have your credentials ready? (y/n): ").lower()
    
    if choice != 'y':
        print("\n📋 Please get your credentials first, then run this script again")
        return False
    
    print()
    client_id = input("Enter your Client ID: ").strip()
    client_secret = input("Enter your Client Secret: ").strip()
    
    if not client_id or not client_secret:
        print("❌ Invalid credentials")
        return False
    
    # Save to .env file
    with open('.env', 'w') as f:
        f.write(f"FATSECRET_CLIENT_ID={client_id}\n")
        f.write(f"FATSECRET_CLIENT_SECRET={client_secret}\n")
    
    # Set for current session
    os.environ['FATSECRET_CLIENT_ID'] = client_id
    os.environ['FATSECRET_CLIENT_SECRET'] = client_secret
    
    print("\n✅ Credentials saved to .env file")
    return True


def test_api_connection():
    """Test connection to FatSecret API"""
    print("\n[STEP 4/6] Testing API connection...")
    
    try:
        from fatsecret_client import FatSecretClient
        
        client = FatSecretClient()
        
        # Try a simple search
        results = client.search_foods('apple', max_results=1)
        
        if results.get('food'):
            print("✅ API connection successful")
            return True
        else:
            print("❌ API returned no results")
            return False
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


def train_models():
    """Train ML models"""
    print("\n[STEP 5/6] Training ML models...")
    print("This may take 5-10 minutes. Please be patient...")
    print()
    
    try:
        from api_model_trainer import train_models_from_api
        
        train_models_from_api()
        
        print("\n✅ Models trained successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scanner():
    """Test the food scanner"""
    print("\n[STEP 6/6] Testing food scanner...")
    
    try:
        from api_food_scanner import ApiFoodScanner
        
        scanner = ApiFoodScanner()
        
        # Try a simple search
        results = scanner.scan_food('salmon', limit=1)
        
        if results:
            result = results[0]
            print(f"\n✅ Scanner test successful!")
            print(f"   Found: {result.name}")
            print(f"   Category: {result.category}")
            print(f"   Calories: {result.nutrients['calories']:.1f} per 100g")
            return True
        else:
            print("❌ Scanner returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup flow"""
    print_header("AI NUTRITION SCANNER - FATSECRET API SETUP")
    
    # Run setup steps
    steps = [
        ("Checking Python", check_python),
        ("Installing packages", install_packages),
        ("Setting up credentials", setup_credentials),
        ("Testing API connection", test_api_connection),
        ("Training ML models", train_models),
        ("Testing scanner", test_scanner)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please fix the issues above and run setup again")
            return False
    
    # Success!
    print_header("✅ SETUP COMPLETE!")
    
    print("Your AI Food Scanner is ready to use!\n")
    print("📁 Files created:")
    print("   - models/food_classifier.pkl")
    print("   - models/scaler.pkl")
    print("   - models/label_encoder.pkl")
    print("   - models/model_metadata.json")
    print("   - models/training_metrics.json")
    print("   - training_data.csv")
    print("   - .env (credentials)")
    
    print("\n📖 Next steps:")
    print("   1. Read API_INTEGRATION_README.md for documentation")
    print("   2. Use api_food_scanner.py in your application")
    print("   3. Replace old food_composition_database.py calls")
    
    print("\n🚀 Quick test:")
    print("   python api_food_scanner.py")
    
    print("\n" + "=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

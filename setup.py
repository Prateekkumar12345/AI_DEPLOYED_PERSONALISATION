#!/usr/bin/env python3
"""
Setup script for AI System with Unified Database
This script helps set up the environment and verify everything works
"""

import os
import sys
from pathlib import Path
import json

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_python_version():
    """Check if Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        'fastapi',
        'uvicorn',
        'streamlit',
        'openai',
        'langchain',
        'langchain_openai',
        'python-dotenv',
        'pydantic'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print("\n❌ Missing dependencies. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def create_directories():
    """Create required directories"""
    print("\nCreating directories...")
    dirs = [
        'shared_data',
        'user_data',
        'user_data/profiles',
        'user_data/reports'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ {dir_path}/")
    
    return True

def check_env_file():
    """Check if .env file exists"""
    print("\nChecking environment configuration...")
    env_path = Path('.env')
    
    if not env_path.exists():
        print("⚠️  .env file not found")
        create = input("Create .env file now? (y/n): ").lower().strip()
        
        if create == 'y':
            api_key = input("Enter your OpenAI API key: ").strip()
            with open('.env', 'w') as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
            print("✅ .env file created")
            return True
        else:
            print("❌ .env file required")
            return False
    
    print("✅ .env file exists")
    
    # Check if API key is set
    with open('.env') as f:
        content = f.read()
        if 'OPENAI_API_KEY' not in content or 'your-api-key' in content.lower():
            print("⚠️  OpenAI API key not properly configured")
            return False
    
    print("✅ OpenAI API key configured")
    return True

def check_required_files():
    """Check if all required Python files exist"""
    print("\nChecking required files...")
    required_files = [
        'shared_database.py',
        'main.py',
        'interviewer.py',
        'personalization_module.py'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            all_exist = False
    
    return all_exist

def initialize_database():
    """Verify database connection to external API"""
    print("\nInitializing database connection...")
    
    try:
        from shared_database import SharedDatabase, EXTERNAL_DB_API_URL
        db = SharedDatabase()
        print("✅ Database client initialized (external API)")
        print(f"   API URL: {EXTERNAL_DB_API_URL}")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def run_health_checks():
    """Run health checks on all components"""
    print("\nRunning health checks...")
    
    # Check if we can import modules
    try:
        from shared_database import SharedDatabase
        print("✅ shared_database module imports correctly")
    except Exception as e:
        print(f"❌ shared_database import error: {e}")
        return False
    
    try:
        # Try to initialize database
        db = SharedDatabase()
        
        # Create a test user
        test_user = db.get_or_create_user("test_user")
        print(f"✅ Test user created: {test_user['username']}")
        
        # Clean up test user (optional)
        # In production, you might want to keep this
        print("✅ All health checks passed")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print("Next steps:")
    print()
    print("1. Start the Academic Chatbot:")
    print("   python main_modified.py")
    print("   Access at: http://localhost:8000")
    print()
    print("2. Start the AI Interviewer:")
    print("   streamlit run interviewer_modified.py")
    print("   Opens automatically in browser")
    print()
    print("3. Start the Personalization Module:")
    print("   python personalization_module_modified.py")
    print("   Access at: http://localhost:8001")
    print()
    print("4. Test the system:")
    print("   - Create a user in any module")
    print("   - Have some interactions")
    print("   - Check personalization analysis")
    print()
    print("📚 For more details, see README.md")
    print()

def main():
    """Main setup routine"""
    print_header("AI System Setup - Unified Database")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Environment File", check_env_file),
        ("Required Files", check_required_files),
        ("Database Initialization", initialize_database),
        ("Health Checks", run_health_checks)
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
                print(f"\n❌ {name} check failed")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {name} check failed with error: {e}")
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n❌ Setup incomplete. Please fix the issues above and run again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
import sys
import subprocess
import traceback

def install_and_run():
    print("=== System Check & Startup ===")
    
    # 1. Check if Flask is installed
    try:
        import flask
        print("✅ Flask is installed.")
    except ImportError:
        print("⚠️ Flask is NOT installed. Installing it automatically now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
            print("✅ Flask installed successfully.")
        except Exception as e:
            print("❌ Failed to install Flask. Please run: pip install flask")
            return
            
    # 2. Try to run the app
    try:
        print("🚀 Starting the Web Dashboard on port 5050...")
        import app
        # Run it on port 5050 just in case 5000 is occupied
        app.app.run(debug=True, port=5050, use_reloader=False)
    except Exception as e:
        print("\n" + "="*50)
        print("❌ CRASH DETECTED IN APP.PY:")
        print("="*50)
        traceback.print_exc()
        print("="*50)
        print("Please copy the exact error message above and show it to the AI.")

if __name__ == "__main__":
    install_and_run()

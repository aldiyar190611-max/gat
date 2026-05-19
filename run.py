import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", "8502", "--server.headless", "true", "--theme.base", "dark"])

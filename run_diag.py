import subprocess
import sys

try:
    result = subprocess.run([sys.executable, 'C:/Users/mdfay/PycharmProjects/Stock_Analysis/diagnose_yfinance.py'], capture_output=True, text=True, check=True)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error running script. Return code: {e.returncode}")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)

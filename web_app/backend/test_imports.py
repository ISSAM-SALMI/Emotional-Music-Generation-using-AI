import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

print("Testing imports...")
try:
    from src import generate, config
    print("Imports successful!")
    print(f"Config ROOT_DIR: {config.ROOT_DIR}")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

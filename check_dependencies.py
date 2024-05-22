import subprocess

# List of packages to check
packages = [
    'streamlit',
    'torch',
    'transformers',
    'requests',
    'sentencepiece',
    'tokenizers'
]

# Function to check if a package is installed
def check_package(package):
    try:
        __import__(package)
        print(f"{package} is installed")
    except ImportError:
        print(f"{package} is NOT installed")

# Check each package
for package in packages:
    check_package(package)

# Print version of transformers to see if it includes required backends
subprocess.run(["pip", "show", "transformers"])

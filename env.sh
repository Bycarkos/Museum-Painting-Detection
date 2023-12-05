#!/usr/bin/bash

# Path to your project directory
PROJECT_DIR="."

# Name of the virtual environment
ENV_NAME="myenv"

# Navigate to the project directory
cd "$PROJECT_DIR"

# Create the virtual environment
python3 -m venv "$ENV_NAME"

# Activate the virtual environment
source "$ENV_NAME/bin/activate"

# Install any required packages in the virtual environment
# For example:
# pip install package1 package2
pip install -r requirements.txt
# Optionally, you can print a message indicating successful activation
echo "Virtual environment $ENV_NAME has been created and activated."

source "$ENV_NAME/bin/activate"

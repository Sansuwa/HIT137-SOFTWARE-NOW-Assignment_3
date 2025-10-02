# HIT137 Assignment 3 - Group Project 

## Sydney Group - 37

## 
## What this package contains
- `main.py` - entrypoint that launches the GUI
- `models.py` - Hugging Face model integration (pipelines)
- `requirements.txt` - Python dependencies


## Setup and run
<!-- venv is required only when python is not globally available for every project. -->
1. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate  # Windows PowerShell
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program:
   ```bash
   python main.py
   ```

## Notes
- On first model usage, Hugging Face will download model weights (internet required).
- If you face issues with heavy models, choose smaller models on https://huggingface.co/models and update `models.py`.
- Make sure to keep your GitHub repository public and add team members before submission.

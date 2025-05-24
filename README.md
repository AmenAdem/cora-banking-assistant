# cora-banking-assistant
CORA is an AI-powered customer support agent designed for intelligent, contextual resolution of customer requests in banking and financial services.

## Startup Guide

1. **Create a virtual environment** (if you haven't already):
   ```powershell
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\Activate
   ```

3. **Install the requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run your project** (replace `main.py` with your actual entry point if different):
   ```powershell
   python main.py
   ```

   If you use uvicorn (common for FastAPI apps), run:
   ```powershell
   uvicorn main:app --reload
   ```
   (Replace `main:app` with the correct module and app variable if needed.)

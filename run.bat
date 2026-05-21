@echo off
echo Starting Finance Audit Agent...
echo.
echo Starting Ollama...
start cmd /k "ollama serve"
timeout /t 3 /nobreak > nul

echo Starting API...
start cmd /k "venv\Scripts\activate && uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 /nobreak > nul

echo Starting Frontend...
start cmd /k "venv\Scripts\activate && streamlit run frontend/app.py"
timeout /t 2 /nobreak > nul

echo.
echo All services started!
echo Open http://localhost:8501 in your browser
pause

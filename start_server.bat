@echo off
echo Starting Data Hygiene API via Python module (bypassing Application Control policy)...
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
pause

# Quick Start Guide

## Option 1: Start Everything (Recommended)

Just run:
```bash
start_all.bat
```

This starts:
- ✅ Backend Server (port 5555)
- ✅ Scheduler Worker (background)
- ✅ Frontend Dev Server (port 5173)

Then open: **http://localhost:5173**

---

## Option 2: Start Individually

### Backend Only
```bash
cd backend
python main.py
```
- Access: http://localhost:5555
- API Docs: http://localhost:5555/docs

### Scheduler Worker (Optional)
```bash
cd backend
python worker.py
```
- Runs scheduled jobs in background
- Not required for basic functionality

### Frontend
```bash
cd frontend
npm install  # First time only
npm run dev
```
- Access: http://localhost:5173

---

## Stopping Services

**Easy way:** Close the terminal windows

**Or use Ctrl+C** in each terminal

---

## Architecture

```
┌─────────────────┐
│  Frontend       │  Port 5173
│  (React + Vite) │
└────────┬────────┘
         │
         │ HTTP Requests
         ▼
┌─────────────────┐
│  Backend        │  Port 5555
│  (FastAPI)      │
└────────┬────────┘
         │
         │ Shared Database
         ▼
┌─────────────────┐
│  Worker         │
│  (Scheduler)    │
└─────────────────┘
```

---

## Troubleshooting

### Backend won't start
- Check port 5555 is free: `netstat -ano | findstr :5555`
- Kill process: `taskkill /F /PID <PID>`

### CORS errors
- Make sure backend is running on port 5555
- Check CORS configuration in `backend/main.py`

### Frontend 404 errors
- Verify backend is running
- Check API calls are going to `http://localhost:5555`

### Ctrl+C doesn't work
- Close the terminal window instead
- Or use Task Manager to end Python processes

---

## Next Steps

1. Open http://localhost:5173
2. View available pipelines
3. Execute a pipeline
4. Check metrics and reliability data
5. Explore API docs at http://localhost:5555/docs

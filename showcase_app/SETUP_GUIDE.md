## IA Modules Showcase App - Setup Guide

Complete setup guide for running the IA Modules showcase application.

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm or yarn
- Git

## Quick Start (Development)

### 1. Backend Setup

```bash
# Navigate to showcase app backend
cd showcase_app/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install IA Modules and dependencies
pip install -e ../..  # Install parent ia_modules package
pip install -r requirements.txt

# Run backend server
python main.py
```

Backend will run on http://localhost:8000

API documentation available at http://localhost:8000/docs

### 2. Frontend Setup

```bash
# In a new terminal, navigate to frontend
cd showcase_app/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run on http://localhost:5173

### 3. Access the Application

Open your browser and navigate to http://localhost:5173

## Environment Variables

### Backend (.env)

Create `backend/.env` file:

```env
# Database (optional, uses in-memory by default)
DATABASE_URL=sqlite:///./metrics.db

# CORS Origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3001

# Logging
LOG_LEVEL=info
DEBUG=true

# LLM API Keys (optional, for AI-powered demos)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### Frontend (.env)

Create `frontend/.env` file:

```env
VITE_API_URL=http://localhost:8000
```

## Docker Deployment

### Build and Run with Docker Compose

```bash
# From showcase_app directory
docker-compose up --build
```

Access at http://localhost:5173

### Individual Docker Commands

**Backend:**
```bash
cd backend
docker build -t ia-modules-backend .
docker run -p 8000:8000 ia-modules-backend
```

**Frontend:**
```bash
cd frontend
docker build -t ia-modules-frontend .
docker run -p 3000:3000 ia-modules-frontend
```

## Production Deployment

### Backend

```bash
# Install production dependencies
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend

```bash
# Build for production
npm run build

# Serve with nginx or any static file server
# Build output is in dist/
```

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Find and kill process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

**Module not found errors:**
```bash
# Ensure IA Modules is installed
pip install -e ../..
```

**Database errors:**
```bash
# Reset database
rm metrics.db
# Restart backend
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Use different port
PORT=3001 npm run dev
```

**API connection errors:**
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check .env has correct API URL
cat .env
```

**Build errors:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### WebSocket Issues

**Connection refused:**
- Check backend WebSocket endpoint is accessible
- Verify firewall settings
- Check browser console for CORS errors

## Features to Try

### 1. Run Example Pipelines
- Navigate to "Pipelines" page
- Click "Execute" on any example pipeline
- Watch real-time execution in "Executions" page

### 2. Monitor Reliability Metrics
- Go to "Metrics" page
- Run several pipelines
- Watch metrics update in real-time
- Check SLO compliance status

### 3. View Execution History
- Navigate to "Executions" page
- See all pipeline runs with status
- Monitor progress bars and durations

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn main:app --reload --port 8000

# Run tests
pytest ../tests/ -v

# Format code
black .
ruff check .
```

### Frontend Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## API Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# List pipelines
curl http://localhost:8000/api/pipelines

# Execute pipeline
curl -X POST http://localhost:8000/api/execute/<pipeline-id> \
  -H "Content-Type: application/json" \
  -d '{"input_data": {}, "checkpoint_enabled": true}'

# Get metrics report
curl http://localhost:8000/api/metrics/report
```

### Using Python

```python
import requests

# List pipelines
response = requests.get("http://localhost:8000/api/pipelines")
pipelines = response.json()
print(f"Found {len(pipelines)} pipelines")

# Execute pipeline
pipeline_id = pipelines[0]["id"]
response = requests.post(
    f"http://localhost:8000/api/execute/{pipeline_id}",
    json={"input_data": {}, "checkpoint_enabled": True}
)
execution = response.json()
print(f"Started execution: {execution['job_id']}")

# Get metrics
response = requests.get("http://localhost:8000/api/metrics/report")
metrics = response.json()
print(f"Success Rate: {metrics['sr']:.2%}")
```

## Performance Tuning

### Backend

- Increase worker processes for production
- Enable Redis for caching (optional)
- Use PostgreSQL for production metrics
- Configure connection pooling

### Frontend

- Enable gzip compression
- Use CDN for static assets
- Implement lazy loading
- Enable service worker for PWA

## Security Considerations

### Production Checklist

- [ ] Change default SECRET_KEY
- [ ] Configure proper CORS origins
- [ ] Enable HTTPS
- [ ] Set up rate limiting
- [ ] Configure API authentication
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Regular dependency updates

## Getting Help

- **Main Documentation**: [../README.md](../README.md)
- **IA Modules Docs**: [../../docs/](../../docs/)
- **API Documentation**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/yourusername/ia_modules/issues

## Next Steps

1. Explore example pipelines
2. Create your own custom pipeline
3. Monitor reliability metrics
4. Review execution history
5. Read the full documentation

---

**Happy showcasing!** 🚀

# IA Modules Dashboard - Frontend

Modern React dashboard for visual pipeline design and real-time monitoring.

## Features

✅ **Pipeline Management** - List, create, edit, delete pipelines
✅ **Real-Time Monitoring** - Live execution tracking with WebSocket
✅ **Live Logs** - Stream logs during execution
✅ **Metrics Dashboard** - Performance and cost tracking
✅ **Plugin Browser** - Discover and view plugins
✅ **Responsive Design** - TailwindCSS styling
✅ **Modern UI** - Clean, professional interface

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first CSS
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Lucide React** - Icon library
- **date-fns** - Date formatting
- **WebSocket API** - Real-time updates

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The app will be available at http://localhost:3000

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── Layout.jsx          # Main layout with sidebar
│   ├── pages/
│   │   ├── PipelineList.jsx    # Pipeline management
│   │   ├── PipelineDesigner.jsx # Create/edit pipelines
│   │   ├── PipelineMonitor.jsx  # Real-time execution viewer
│   │   ├── MetricsDashboard.jsx # Metrics and charts
│   │   └── PluginsBrowser.jsx   # Plugin discovery
│   ├── services/
│   │   ├── api.js              # REST API client
│   │   └── websocket.js        # WebSocket client
│   ├── App.jsx                 # Main app component
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Pages

### 1. Pipeline List (`/pipelines`)

- View all pipelines
- Search and filter
- Execute, edit, delete pipelines
- Real-time stats (total pipelines, active executions, etc.)

### 2. Pipeline Designer (`/pipelines/new`, `/pipelines/:id/edit`)

- Create new pipelines
- Edit pipeline configuration (JSON editor)
- Save and execute
- **Coming Soon**: Visual drag-and-drop designer with React Flow

### 3. Pipeline Monitor (`/monitor/:executionId`)

- Real-time execution tracking
- Live progress updates
- Step-by-step visualization
- Live log streaming
- Performance metrics (duration, items processed, cost)
- Output and error display

### 4. Metrics Dashboard (`/metrics`)

- **Coming Soon**: Performance charts with Chart.js
- Execution trends
- Cost analysis
- Resource usage

### 5. Plugins Browser (`/plugins`)

- View all available plugins
- Plugin metadata (name, version, type, description, author)
- **Coming Soon**: Install/uninstall plugins

## API Integration

The frontend communicates with the backend API:

### REST API

```javascript
import { pipelines, executions, metrics, plugins } from '@/services/api'

// List pipelines
const response = await pipelines.list({ search: 'query' })

// Execute pipeline
const result = await pipelines.execute(pipelineId, { input: 'data' })

// Get execution status
const status = await executions.getStatus(executionId)
```

### WebSocket

```javascript
import WebSocketService from '@/services/websocket'

const ws = new WebSocketService()
ws.connect(executionId)

// Listen for events
ws.on('execution_started', (message) => {
  console.log('Execution started:', message)
})

ws.on('progress_update', (message) => {
  console.log('Progress:', message.data.progress_percent)
})

ws.on('execution_completed', (message) => {
  console.log('Completed:', message)
})
```

## WebSocket Events

The real-time monitor listens for these events:

- `execution_started` - Pipeline execution begins
- `step_started` - Individual step starts
- `step_completed` - Step completes successfully
- `step_failed` - Step fails with error
- `log_message` - Log entry (debug, info, warning, error)
- `progress_update` - Progress percentage and metrics
- `metrics_update` - Performance metrics update
- `execution_completed` - Pipeline completes successfully
- `execution_failed` - Pipeline fails with error

## Development

### Running Locally

```bash
# Terminal 1: Start backend API
cd ia_modules/dashboard
python run_dashboard.py

# Terminal 2: Start frontend
cd ia_modules/dashboard/frontend
npm run dev
```

Access the dashboard at http://localhost:3000

The Vite dev server proxies API requests to the backend:
- `/api/*` → `http://localhost:8000/api/*`
- `/ws/*` → `ws://localhost:8000/ws/*`

### Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

### Adding New Pages

1. Create component in `src/pages/`:
```jsx
export default function MyPage() {
  return <div>My Page</div>
}
```

2. Add route in `src/App.jsx`:
```jsx
<Route path="/my-page" element={<MyPage />} />
```

3. Add navigation in `src/components/Layout.jsx`:
```jsx
{ name: 'My Page', href: '/my-page', icon: Icon }
```

## Styling

### TailwindCSS Utilities

Common classes used:

```jsx
// Buttons
<button className="btn-primary">Primary</button>
<button className="btn-secondary">Secondary</button>

// Cards
<div className="card">Content</div>

// Grid layouts
<div className="grid grid-cols-3 gap-4">...</div>

// Flex layouts
<div className="flex items-center justify-between">...</div>
```

### Custom Components

Reusable components can be added to `src/components/`:

```jsx
// src/components/Button.jsx
export default function Button({ children, ...props }) {
  return (
    <button className="btn-primary" {...props}>
      {children}
    </button>
  )
}
```

## Deployment

### Build for Production

```bash
npm run build
```

This creates optimized files in `dist/`:

```
dist/
├── index.html
├── assets/
│   ├── index-[hash].js
│   └── index-[hash].css
└── ...
```

### Deploy with Docker

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx Configuration

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
    }

    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Roadmap

### Phase 1: Core Features ✅ (Current)
- [x] Pipeline list and management
- [x] Real-time execution monitoring
- [x] WebSocket integration
- [x] Live logs
- [x] Basic metrics display
- [x] Plugin browser

### Phase 2: Visual Designer (Week 3)
- [ ] React Flow integration
- [ ] Drag-and-drop step creator
- [ ] Visual flow connectors
- [ ] Step configuration forms
- [ ] Condition builder UI
- [ ] JSON preview sync

### Phase 3: Enhanced Metrics (Week 3)
- [ ] Chart.js integration
- [ ] Performance charts
- [ ] Cost analysis graphs
- [ ] Resource usage visualization
- [ ] Historical trends

### Phase 4: Advanced Features (Week 4)
- [ ] Pipeline debugger (step-by-step)
- [ ] Variable inspection
- [ ] Breakpoints
- [ ] Mock data injection
- [ ] Dark mode
- [ ] User authentication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Troubleshooting

### API Connection Issues

If you can't connect to the API:

1. Check backend is running: `curl http://localhost:8000/health`
2. Check proxy configuration in `vite.config.js`
3. Check CORS settings in backend

### WebSocket Connection Fails

1. Check WebSocket URL in browser console
2. Verify backend WebSocket endpoint is running
3. Check firewall/proxy settings

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

## License

Part of IA Modules - See main LICENSE file

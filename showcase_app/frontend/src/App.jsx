import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom'
import { Home, BarChart3, Play, FileCode, Github, Edit, Sparkles, Network, Moon, Sun, HelpCircle, Menu, X, Database } from 'lucide-react'
import { Toaster } from 'react-hot-toast'
import axios from 'axios'
import HomePage from './pages/HomePage'
import PipelinesPage from './pages/PipelinesPage'
import MetricsPage from './pages/MetricsPage'
import ExecutionsPage from './pages/ExecutionsPage'
import ExecutionDetailPage from './pages/ExecutionDetailPage'
import PipelineEditorPage from './pages/PipelineEditorPage'
import PatternsPage from './pages/PatternsPage'
import MultiAgentDashboard from './components/MultiAgent/MultiAgentDashboard'
import ErrorBoundary from './components/ErrorBoundary/ErrorBoundary'
import ThemeProvider, { useTheme } from './components/ThemeProvider/ThemeProvider'
import KeyboardShortcutsModal from './components/ui/keyboard-shortcuts-modal'
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts'

function AppContent() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [shortcutsModalOpen, setShortcutsModalOpen] = useState(false)
  const [backendStatus, setBackendStatus] = useState({ 
    connected: false, 
    database: 'checking...',
    websocket: false 
  })
  const { theme, toggleTheme } = useTheme()
  const location = useLocation()

  // Check backend status
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5555'
        const response = await axios.get(`${apiUrl}/health`, {
          timeout: 5000,
          headers: { 'Accept': 'application/json' }
        })
        setBackendStatus({
          connected: true,
          database: response.data.database || 'unknown',
          websocket: false // Will be updated by websocket connection
        })
      } catch (error) {
        console.error('Backend health check failed:', error.message)
        setBackendStatus({ 
          connected: false, 
          database: 'disconnected',
          websocket: false 
        })
      }
    }
    
    checkBackend()
    const interval = setInterval(checkBackend, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!backendStatus.connected) return

    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:5555'
    const ws = new WebSocket(`${wsUrl}/ws/metrics`)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      setBackendStatus(prev => ({ ...prev, websocket: true }))
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setBackendStatus(prev => ({ ...prev, websocket: false }))
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setBackendStatus(prev => ({ ...prev, websocket: false }))
    }

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [backendStatus.connected])

  // Keyboard shortcuts
  useKeyboardShortcuts({
    'cmd+/': () => setShortcutsModalOpen(true),
    'cmd+b': () => setSidebarOpen(prev => !prev),
    'cmd+d': () => toggleTheme(),
    'esc': () => {
      setShortcutsModalOpen(false)
      setMobileMenuOpen(false)
    },
  })

  const shortcuts = {
    'Navigation': [
      { keys: 'cmd+/', description: 'Show keyboard shortcuts' },
      { keys: 'cmd+b', description: 'Toggle sidebar' },
      { keys: 'esc', description: 'Close modals' },
    ],
    'Appearance': [
      { keys: 'cmd+d', description: 'Toggle dark mode' },
    ],
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Mobile Menu Button */}
      <button
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-gray-900 dark:bg-gray-800 text-white rounded-lg"
      >
        {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Mobile Menu Backdrop */}
      {mobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        bg-gray-900 dark:bg-gray-950 text-white 
        ${sidebarOpen ? 'w-64' : 'w-20'} 
        transition-all duration-300 flex flex-col
        fixed lg:relative inset-y-0 left-0 z-40
        ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-4 border-b border-gray-700 dark:border-gray-800">
          <div className="flex items-center space-x-2">
            <div className="bg-primary-500 rounded p-2">
              <FileCode size={24} />
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="font-bold text-lg">IA Modules</h1>
                <p className="text-xs text-gray-400">Showcase v0.0.3</p>
              </div>
            )}
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          <NavLink to="/" icon={<Home size={20} />} text="Home" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
          <NavLink to="/pipelines" icon={<FileCode size={20} />} text="Pipelines" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
          <NavLink to="/editor" icon={<Edit size={20} />} text="Editor" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
          <NavLink to="/executions" icon={<Play size={20} />} text="Executions" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
          <NavLink to="/patterns" icon={<Sparkles size={20} />} text="Patterns" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
          <NavLink to="/multi-agent" icon={<Network size={20} />} text="Multi-Agent" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
          <NavLink to="/metrics" icon={<BarChart3 size={20} />} text="Metrics" sidebarOpen={sidebarOpen} onClick={() => setMobileMenuOpen(false)} />
        </nav>

        <div className="p-4 border-t border-gray-700 dark:border-gray-800 space-y-2">
          <button
            onClick={() => setShortcutsModalOpen(true)}
            className="w-full flex items-center space-x-2 px-4 py-3 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-900 transition text-gray-400 hover:text-white"
          >
            <HelpCircle size={20} />
            {sidebarOpen && <span className="text-sm">Shortcuts</span>}
          </button>
          <a
            href="https://github.com/yourusername/ia_modules"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 px-4 py-3 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-900 transition text-gray-400 hover:text-white"
          >
            <Github size={20} />
            {sidebarOpen && <span className="text-sm">GitHub</span>}
          </a>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 shadow-sm px-4 lg:px-6 py-4 flex items-center justify-between border-b border-gray-200 dark:border-gray-700">
          <div className="ml-12 lg:ml-0">
            <h2 className="text-xl lg:text-2xl font-bold text-gray-800 dark:text-gray-100">IA Modules Showcase</h2>
            <p className="text-xs lg:text-sm text-gray-600 dark:text-gray-400">Production-ready AI agent framework with enterprise-grade reliability</p>
          </div>
          <div className="flex items-center space-x-3">
            <div className="hidden sm:flex items-center space-x-4 text-sm">
              {/* Database Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${backendStatus.connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                <Database size={16} className="text-gray-600 dark:text-gray-400" />
                <span className="text-gray-600 dark:text-gray-400">
                  {backendStatus.connected ? backendStatus.database : 'Disconnected'}
                </span>
              </div>
              
              {/* WebSocket Status */}
              {backendStatus.connected && (
                <div className="flex items-center space-x-2 pl-4 border-l border-gray-300 dark:border-gray-600">
                  <div className={`w-2 h-2 rounded-full ${backendStatus.websocket ? 'bg-blue-500 animate-pulse' : 'bg-gray-400'}`}></div>
                  <span className="text-gray-600 dark:text-gray-400">
                    WS {backendStatus.websocket ? 'Connected' : 'Offline'}
                  </span>
                </div>
              )}
            </div>
            <button
              onClick={toggleTheme}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
              aria-label="Toggle dark mode"
            >
              {theme === 'light' ? (
                <Moon size={20} className="text-gray-600 dark:text-gray-400" />
              ) : (
                <Sun size={20} className="text-gray-400" />
              )}
            </button>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-4 lg:p-6">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/pipelines" element={<PipelinesPage />} />
            <Route path="/editor" element={<PipelineEditorPage />} />
            <Route path="/executions" element={<ExecutionsPage />} />
            <Route path="/executions/:jobId" element={<ExecutionDetailPage />} />
            <Route path="/patterns" element={<PatternsPage />} />
            <Route path="/multi-agent" element={<MultiAgentDashboard />} />
            <Route path="/metrics" element={<MetricsPage />} />
          </Routes>
        </main>
      </div>

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        isOpen={shortcutsModalOpen}
        onClose={() => setShortcutsModalOpen(false)}
        shortcuts={shortcuts}
      />

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: theme === 'dark' ? '#1f2937' : '#fff',
            color: theme === 'dark' ? '#f9fafb' : '#111827',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </div>
  )
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="light">
        <BrowserRouter>
          <AppContent />
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  )
}

function NavLink({ to, icon, text, sidebarOpen, onClick }) {
  const location = useLocation()
  const isActive = location.pathname === to || (to !== '/' && location.pathname.startsWith(to))

  return (
    <Link
      to={to}
      onClick={onClick}
      className={`
        flex items-center space-x-3 px-4 py-3 rounded-lg transition group
        ${isActive 
          ? 'bg-primary-600 text-white' 
          : 'hover:bg-gray-800 dark:hover:bg-gray-900 text-gray-400 hover:text-white'
        }
      `}
    >
      <div className={isActive ? 'text-white' : 'text-gray-400 group-hover:text-white'}>{icon}</div>
      {sidebarOpen && <span className={isActive ? 'text-white' : 'group-hover:text-white'}>{text}</span>}
    </Link>
  )
}

export default App

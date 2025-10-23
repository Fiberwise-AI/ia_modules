import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import { Home, BarChart3, Play, FileCode, Github } from 'lucide-react'
import HomePage from './pages/HomePage'
import PipelinesPage from './pages/PipelinesPage'
import MetricsPage from './pages/MetricsPage'
import ExecutionsPage from './pages/ExecutionsPage'
import ExecutionDetailPage from './pages/ExecutionDetailPage'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <aside className={`bg-gray-900 text-white ${sidebarOpen ? 'w-64' : 'w-20'} transition-all duration-300 flex flex-col`}>
          <div className="p-4 border-b border-gray-700">
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

          <nav className="flex-1 p-4 space-y-2">
            <NavLink to="/" icon={<Home size={20} />} text="Home" sidebarOpen={sidebarOpen} />
            <NavLink to="/pipelines" icon={<FileCode size={20} />} text="Pipelines" sidebarOpen={sidebarOpen} />
            <NavLink to="/executions" icon={<Play size={20} />} text="Executions" sidebarOpen={sidebarOpen} />
            <NavLink to="/metrics" icon={<BarChart3 size={20} />} text="Metrics" sidebarOpen={sidebarOpen} />
          </nav>

          <div className="p-4 border-t border-gray-700">
            <a
              href="https://github.com/yourusername/ia_modules"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-gray-400 hover:text-white transition"
            >
              <Github size={20} />
              {sidebarOpen && <span className="text-sm">GitHub</span>}
            </a>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="bg-white shadow-sm px-6 py-4 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-800">IA Modules Showcase</h2>
              <p className="text-sm text-gray-600">Production-ready AI agent framework with enterprise-grade reliability</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-gray-600">Backend Connected</span>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 overflow-auto p-6">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/pipelines" element={<PipelinesPage />} />
              <Route path="/executions" element={<ExecutionsPage />} />
              <Route path="/executions/:jobId" element={<ExecutionDetailPage />} />
              <Route path="/metrics" element={<MetricsPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  )
}

function NavLink({ to, icon, text, sidebarOpen }) {
  return (
    <Link
      to={to}
      className="flex items-center space-x-3 px-4 py-3 rounded-lg hover:bg-gray-800 transition group"
    >
      <div className="text-gray-400 group-hover:text-white">{icon}</div>
      {sidebarOpen && <span className="group-hover:text-white">{text}</span>}
    </Link>
  )
}

export default App

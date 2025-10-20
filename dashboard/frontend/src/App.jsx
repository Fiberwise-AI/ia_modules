import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import PipelineList from './pages/PipelineList'
import PipelineDesigner from './pages/PipelineDesigner'
import PipelineMonitor from './pages/PipelineMonitor'
import MetricsDashboard from './pages/MetricsDashboard'
import PluginsBrowser from './pages/PluginsBrowser'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<PipelineList />} />
        <Route path="/pipelines" element={<PipelineList />} />
        <Route path="/pipelines/new" element={<PipelineDesigner />} />
        <Route path="/pipelines/:id/edit" element={<PipelineDesigner />} />
        <Route path="/monitor/:executionId" element={<PipelineMonitor />} />
        <Route path="/metrics" element={<MetricsDashboard />} />
        <Route path="/plugins" element={<PluginsBrowser />} />
      </Routes>
    </Layout>
  )
}

export default App

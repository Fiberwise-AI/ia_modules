import { useState, useEffect } from 'react'
import { BarChart3 } from 'lucide-react'
import { metrics as metricsApi } from '../services/api'

export default function MetricsDashboard() {
  const [metricsData, setMetricsData] = useState(null)

  useEffect(() => {
    loadMetrics()
  }, [])

  async function loadMetrics() {
    try {
      const response = await metricsApi.get({ time_range: '1h' })
      setMetricsData(response.data)
    } catch (error) {
      console.error('Failed to load metrics:', error)
    }
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Metrics Dashboard</h1>

      <div className="card text-center py-12">
        <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Charts Coming Soon
        </h2>
        <p className="text-gray-500">
          Real-time metrics visualization with Chart.js will be added in the next update.
        </p>
        <div className="mt-4">
          <a
            href="/api/metrics/prometheus"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-600 hover:text-primary-700"
          >
            View Prometheus Metrics →
          </a>
        </div>
      </div>
    </div>
  )
}

import axios from 'axios'

// Use environment variable or default to backend directly
// In development, call backend directly (CORS is configured)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5555/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Include credentials for CORS
})

// Pipelines
export const pipelinesAPI = {
  list: () => api.get('/pipelines'),
  get: (id) => api.get(`/pipelines/${id}`),
  create: (data) => api.post('/pipelines', data),
  update: (id, data) => api.put(`/pipelines/${id}`, data),
  delete: (id) => api.delete(`/pipelines/${id}`),
}

// Execution
export const executionAPI = {
  start: (pipelineId, inputData) =>
    api.post(`/execute/${pipelineId}`, {
      input_data: inputData,
      checkpoint_enabled: true,
    }),
  get: (jobId) => api.get(`/execute/${jobId}`),
  list: () => api.get('/execute'),
  cancel: (jobId) => api.delete(`/execute/${jobId}`),
}

// Metrics
export const metricsAPI = {
  getReport: () => api.get('/metrics/report'),
  getSLO: () => api.get('/metrics/slo'),
  getEvents: (limit = 100) => api.get(`/metrics/events?limit=${limit}`),
  getHistory: (hours = 24) => api.get(`/metrics/history?hours=${hours}`),
}

// Health
export const healthAPI = {
  check: () => axios.get('http://localhost:5555/health'),
}

export default api

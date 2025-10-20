import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Pipelines
export const pipelines = {
  list: (params) => api.get('/pipelines', { params }),
  get: (id) => api.get(`/pipelines/${id}`),
  create: (data) => api.post('/pipelines', data),
  update: (id, data) => api.put(`/pipelines/${id}`, data),
  delete: (id) => api.delete(`/pipelines/${id}`),
  validate: (config) => api.post('/pipelines/validate', config),
  execute: (id, inputData) => api.post(`/pipelines/${id}/execute`, { input_data: inputData }),
}

// Executions
export const executions = {
  getStatus: (id) => api.get(`/executions/${id}/status`),
  getLogs: (id) => api.get(`/executions/${id}/logs`),
  cancel: (id) => api.post(`/executions/${id}/cancel`),
}

// Metrics
export const metrics = {
  get: (params) => api.get('/metrics', { params }),
  prometheus: () => api.get('/metrics/prometheus'),
  benchmarks: (params) => api.get('/benchmarks', { params }),
}

// Plugins
export const plugins = {
  list: () => api.get('/plugins'),
}

// System
export const system = {
  health: () => api.get('/health'),
  stats: () => api.get('/stats'),
}

export default api

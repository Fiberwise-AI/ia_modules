import axios from 'axios'
import toast from 'react-hot-toast'

// Use environment variable for API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5555'

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Include credentials for CORS
})

// Request interceptor for loading state
api.interceptors.request.use(
  (config) => {
    // Add a loading toast ID to the config for tracking
    if (config.showLoading !== false) {
      config.loadingToastId = toast.loading(config.loadingMessage || 'Loading...', {
        duration: Infinity,
      })
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    // Dismiss loading toast
    if (response.config.loadingToastId) {
      toast.dismiss(response.config.loadingToastId)
    }
    
    // Show success toast if configured
    if (response.config.successMessage) {
      toast.success(response.config.successMessage)
    }
    
    return response
  },
  (error) => {
    // Dismiss loading toast
    if (error.config?.loadingToastId) {
      toast.dismiss(error.config.loadingToastId)
    }
    
    // Show error toast with appropriate message
    const errorMessage = 
      error.response?.data?.error || 
      error.response?.data?.message || 
      error.message || 
      'An unexpected error occurred'
    
    // Don't show toast if explicitly disabled
    if (error.config?.showError !== false) {
      toast.error(errorMessage, {
        duration: 5000,
      })
    }
    
    return Promise.reject(error)
  }
)

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

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { execSync } from 'child_process'

// Resolve WSL2 IP so the Vite proxy (Windows) can reach the backend (WSL2).
// Falls back to localhost for native setups.
let backendHost = 'localhost'
try {
  const wslIp = execSync('wsl -d Ubuntu bash -lc "hostname -I"', { encoding: 'utf-8' }).trim().split(/\s+/)[0]
  if (wslIp) backendHost = wslIp
} catch {
  // Not running under WSL or wsl command unavailable — use localhost
}

const API_TARGET = `http://${backendHost}:7331`
const WS_TARGET = `ws://${backendHost}:7331`

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
      '/health': {
        target: API_TARGET,
        changeOrigin: true,
      },
      '/ws': {
        target: WS_TARGET,
        ws: true,
      },
    },
  },
})

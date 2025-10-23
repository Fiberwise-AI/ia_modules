import { useEffect, useRef, useState } from 'react'

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:5555'

export function useExecutionWebSocket(jobId, onUpdate) {
  const ws = useRef(null)
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    if (!jobId) return

    const wsUrl = `${WS_BASE_URL}/ws/execution/${jobId}`
    console.log('Connecting to WebSocket:', wsUrl)

    ws.current = new WebSocket(wsUrl)

    ws.current.onopen = () => {
      console.log('WebSocket connected for execution:', jobId)
      setIsConnected(true)
    }

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('WebSocket message received:', data)
        if (onUpdate) {
          onUpdate(data)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.current.onclose = () => {
      console.log('WebSocket disconnected for execution:', jobId)
      setIsConnected(false)
    }

    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [jobId, onUpdate])

  return { isConnected }
}

export function useMetricsWebSocket(onUpdate) {
  const ws = useRef(null)
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const wsUrl = `${WS_BASE_URL}/ws/metrics`
    console.log('Connecting to WebSocket:', wsUrl)

    ws.current = new WebSocket(wsUrl)

    ws.current.onopen = () => {
      console.log('WebSocket connected for metrics')
      setIsConnected(true)
    }

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('Metrics WebSocket message received:', data)
        if (onUpdate) {
          onUpdate(data)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.current.onerror = (error) => {
      console.error('Metrics WebSocket error:', error)
    }

    ws.current.onclose = () => {
      console.log('Metrics WebSocket disconnected')
      setIsConnected(false)
    }

    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [onUpdate])

  return { isConnected }
}

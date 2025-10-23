import React from 'react'
import { Handle, Position } from 'reactflow'

export default function StepNode({ data }) {
  const statusColor = {
    pending: 'bg-gray-100 border-gray-300',
    running: 'bg-blue-100 border-blue-500 animate-pulse',
    completed: 'bg-green-100 border-green-500',
    failed: 'bg-red-100 border-red-500',
  }[data.status || 'pending']

  return (
    <>
      <Handle type="target" position={Position.Top} />
      <div className={`px-4 py-2 shadow-md rounded-lg border-2 ${statusColor} min-w-[150px]`}>
        <div className="font-bold text-sm">{data.label}</div>
        {data.duration && (
          <div className="text-xs text-gray-600 mt-1">{data.duration}ms</div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </>
  )
}

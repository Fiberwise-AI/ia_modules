import React from 'react'
import { Handle, Position } from 'reactflow'

export default function ParallelNode({ data }) {
  return (
    <>
      <Handle type="target" position={Position.Top} />
      <div className="px-3 py-2 bg-purple-100 border-2 border-purple-500 rounded-lg shadow-md">
        <div className="font-bold text-xs">⚡ {data.label}</div>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </>
  )
}

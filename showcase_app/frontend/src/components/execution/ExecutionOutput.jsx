import React from 'react'

export default function ExecutionOutput({ outputData }) {
  if (!outputData) {
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-4">Final Output</h2>
      <pre className="bg-gray-50 p-4 rounded overflow-x-auto text-sm">
        {JSON.stringify(outputData, null, 2)}
      </pre>
    </div>
  )
}

import React from 'react'

export default function DataViewer({ title, data, maxHeight }) {
  if (!data) return null

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
      </div>
      <div className="p-6">
        <pre className={`text-sm text-gray-900 bg-gray-50 p-4 rounded border overflow-x-auto ${maxHeight || ''}`}>
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    </div>
  )
}

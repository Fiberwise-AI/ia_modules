import React from 'react'
import { ArrowLeft } from 'lucide-react'

export default function ExecutionHeader({ onBack }) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft size={20} />
          <span>Back to Executions</span>
        </button>
      </div>
    </div>
  )
}

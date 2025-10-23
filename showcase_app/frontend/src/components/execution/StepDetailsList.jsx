import React from 'react'
import StepDetailCard from './StepDetailCard'

export default function StepDetailsList({ steps }) {
  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800">Step Execution Details</h2>
        <p className="text-sm text-gray-600 mt-1">View input/output data flow for each step</p>
      </div>
      <div className="p-6">
        {steps && steps.length > 0 ? (
          <div className="space-y-4">
            {steps.map((step, index) => (
              <StepDetailCard key={index} step={step} />
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            No step execution data available
          </div>
        )}
      </div>
    </div>
  )
}

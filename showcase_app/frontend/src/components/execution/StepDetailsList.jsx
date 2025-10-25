import React, { useState } from 'react'
import StepDetailCard from './StepDetailCard'
import StepDetailPanel from './StepDetailPanel'

export default function StepDetailsList({ steps }) {
  const [selectedStep, setSelectedStep] = useState(null)

  return (
    <>
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Step Execution Details</h2>
          <p className="text-sm text-gray-600 mt-1">Click on a step to view detailed information</p>
        </div>
        <div className="p-6">
          {steps && steps.length > 0 ? (
            <div className="space-y-4">
              {steps.map((step, index) => (
                <div key={index} className="relative">
                  <StepDetailCard step={step} />
                  <div className="absolute top-4 right-4">
                    <button
                      onClick={() => setSelectedStep(step)}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg shadow-sm transition-colors flex items-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                      View Details
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No step execution data available
            </div>
          )}
        </div>
      </div>

      {selectedStep && (
        <StepDetailPanel
          step={selectedStep}
          onClose={() => setSelectedStep(null)}
        />
      )}
    </>
  )
}

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
                <div
                  key={index}
                  onClick={() => setSelectedStep(step)}
                  className="cursor-pointer transform transition-transform hover:scale-[1.01]"
                >
                  <StepDetailCard step={step} />
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

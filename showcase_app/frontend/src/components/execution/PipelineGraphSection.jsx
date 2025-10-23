import React from 'react'
import PipelineGraph from '../graph/PipelineGraph'
import PipelineFlowDiagram from './PipelineFlowDiagram'

export default function PipelineGraphSection({ pipeline, execution }) {
  // Show interactive graph if pipeline config available
  if (pipeline) {
    return (
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Pipeline Graph</h2>
          <p className="text-sm text-gray-600">Interactive visualization showing pipeline structure and execution</p>
        </div>
        <div className="p-6">
          <PipelineGraph pipeline={pipeline} execution={execution} />
        </div>
      </div>
    )
  }

  // Fallback to simple flow diagram
  if (execution.steps && execution.steps.length > 0) {
    return (
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Pipeline Flow</h2>
          <p className="text-sm text-gray-600">Sequential step execution</p>
        </div>
        <div className="p-8">
          <PipelineFlowDiagram steps={execution.steps} />
        </div>
      </div>
    )
  }

  return null
}

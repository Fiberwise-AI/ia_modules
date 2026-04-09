import React from 'react'
import PipelineGraph from '../graph/PipelineGraph'
import PipelineFlowDiagram from './PipelineFlowDiagram'

export default function PipelineGraphSection({ pipeline, execution }) {
  // Show interactive graph if pipeline config available
  if (pipeline) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">Pipeline Graph</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">Interactive visualization showing pipeline structure and execution</p>
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
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
          <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">Pipeline Flow</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">Sequential step execution</p>
        </div>
        <div className="p-8">
          <PipelineFlowDiagram steps={execution.steps} />
        </div>
      </div>
    )
  }

  return null
}

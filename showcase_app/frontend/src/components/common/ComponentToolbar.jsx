import React from 'react'
import { Save, RotateCcw, Plus } from 'lucide-react'

/**
 * ComponentToolbar - Toolbar for managing drag-drop components
 * Provides buttons for adding components, saving pipelines, and resetting layout
 */
export default function ComponentToolbar({
  availableComponents = [],
  onAddComponent,
  onSavePipeline,
  onResetLayout
}) {
  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800">Customize Layout</h3>
        <div className="flex space-x-2">
          <button
            onClick={onSavePipeline}
            className="flex items-center px-3 py-1 bg-purple-100 hover:bg-purple-200 text-purple-800 rounded text-sm transition-colors"
            title="Save current layout as a new pipeline"
          >
            <Save className="w-4 h-4 mr-1" />
            Save as Pipeline
          </button>
          <button
            onClick={onResetLayout}
            className="flex items-center px-3 py-1 bg-red-100 hover:bg-red-200 text-red-800 rounded text-sm transition-colors"
            title="Reset to default layout"
          >
            <RotateCcw className="w-4 h-4 mr-1" />
            Reset Layout
          </button>
        </div>
      </div>

      <div className="mb-2">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Add Components:</h4>
        <div className="flex flex-wrap gap-2">
          {availableComponents.map(componentType => (
            <button
              key={componentType}
              onClick={() => onAddComponent(componentType)}
              className="flex items-center px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded text-sm transition-colors"
              title={`Add ${componentType} component`}
            >
              <Plus className="w-3 h-3 mr-1" />
              {componentType}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
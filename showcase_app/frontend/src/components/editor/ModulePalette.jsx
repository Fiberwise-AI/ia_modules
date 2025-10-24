import { Box, Circle, GitBranch, Play, Database, MessageSquare, Code } from 'lucide-react';

const moduleCategories = [
  {
    name: 'Basic Steps',
    modules: [
      { type: 'task', label: 'Task Step', icon: Play, description: 'Generic task execution' },
      { type: 'transform', label: 'Transform', icon: Code, description: 'Data transformation' },
      { type: 'validation', label: 'Validation', icon: Circle, description: 'Data validation' },
    ],
  },
  {
    name: 'Data Operations',
    modules: [
      { type: 'database', label: 'Database', icon: Database, description: 'Database operations' },
      { type: 'api', label: 'API Call', icon: MessageSquare, description: 'External API calls' },
    ],
  },
  {
    name: 'Control Flow',
    modules: [
      { type: 'decision', label: 'Decision', icon: GitBranch, description: 'Conditional branching' },
      { type: 'parallel', label: 'Parallel', icon: Box, description: 'Parallel execution' },
    ],
  },
];

export default function ModulePalette({ onAddStep }) {
  return (
    <div className="w-64 bg-gray-50 border-r p-4 overflow-y-auto">
      <h3 className="font-semibold text-gray-700 mb-4">Module Palette</h3>

      <div className="space-y-4">
        {moduleCategories.map((category) => (
          <div key={category.name}>
            <h4 className="text-sm font-medium text-gray-600 mb-2">{category.name}</h4>
            <div className="space-y-2">
              {category.modules.map((module) => (
                <ModuleCard key={module.type} module={module} onAdd={() => onAddStep(module.type)} />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-xs text-blue-700">
          <strong>Tip:</strong> Drag modules onto the canvas or click to add. Connect nodes by dragging from one
          node's edge to another.
        </p>
      </div>
    </div>
  );
}

function ModuleCard({ module, onAdd }) {
  const Icon = module.icon;

  return (
    <button
      onClick={onAdd}
      className="w-full p-3 bg-white border rounded-lg hover:border-blue-500 hover:shadow-md transition-all text-left group"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 bg-blue-50 rounded group-hover:bg-blue-100 transition-colors">
          <Icon className="w-4 h-4 text-blue-600" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm text-gray-900">{module.label}</div>
          <div className="text-xs text-gray-500 mt-0.5">{module.description}</div>
        </div>
      </div>
    </button>
  );
}

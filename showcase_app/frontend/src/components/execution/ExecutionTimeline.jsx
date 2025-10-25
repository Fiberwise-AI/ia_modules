import { useMemo } from 'react';
import { Gantt, ViewMode } from 'gantt-task-react';
import 'gantt-task-react/dist/index.css';
import { Clock, Activity, CheckCircle, XCircle, Loader } from 'lucide-react';

export default function ExecutionTimeline({ execution }) {
  const { tasks, metrics } = useMemo(() => {
    if (!execution?.steps) {
      return { tasks: [], metrics: {} };
    }

    const steps = execution.steps || [];
    const startTime = new Date(execution.start_time || Date.now());
    
    // Convert steps to Gantt tasks
    const ganttTasks = steps.map((step, index) => {
      const stepStart = step.start_time ? new Date(step.start_time) : startTime;
      const stepEnd = step.end_time 
        ? new Date(step.end_time) 
        : step.status === 'running' 
          ? new Date() 
          : stepStart;
      
      // Ensure end is after start
      const duration = Math.max(stepEnd - stepStart, 1000); // At least 1 second
      const actualEnd = new Date(stepStart.getTime() + duration);

      return {
        id: step.name || `step-${index}`,
        name: step.name || `Step ${index + 1}`,
        start: stepStart,
        end: actualEnd,
        progress: getStepProgress(step),
        type: 'task',
        styles: getTaskStyles(step.status),
        hideChildren: false
      };
    });

    // Calculate metrics
    const completedSteps = steps.filter(s => s.status === 'completed').length;
    const failedSteps = steps.filter(s => s.status === 'failed').length;
    const runningSteps = steps.filter(s => s.status === 'running').length;
    const totalDuration = execution.end_time && execution.start_time
      ? new Date(execution.end_time) - new Date(execution.start_time)
      : Date.now() - new Date(execution.start_time || Date.now());

    return {
      tasks: ganttTasks,
      metrics: {
        total: steps.length,
        completed: completedSteps,
        failed: failedSteps,
        running: runningSteps,
        duration: totalDuration
      }
    };
  }, [execution]);

  if (!execution) {
    return (
      <div className="p-8 text-center text-gray-500">
        <Clock className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p>No execution data available</p>
      </div>
    );
  }

  if (tasks.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p>No steps to display</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200">
      {/* Header with Metrics */}
      <div className="p-4 border-b bg-gray-50">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Execution Timeline
        </h2>

        <div className="grid grid-cols-5 gap-4">
          <MetricCard
            icon={<Activity className="w-4 h-4" />}
            label="Total Steps"
            value={metrics.total}
            color="text-blue-600"
          />
          <MetricCard
            icon={<CheckCircle className="w-4 h-4" />}
            label="Completed"
            value={metrics.completed}
            color="text-green-600"
          />
          <MetricCard
            icon={<XCircle className="w-4 h-4" />}
            label="Failed"
            value={metrics.failed}
            color="text-red-600"
          />
          <MetricCard
            icon={<Loader className="w-4 h-4" />}
            label="Running"
            value={metrics.running}
            color="text-yellow-600"
          />
          <MetricCard
            icon={<Clock className="w-4 h-4" />}
            label="Duration"
            value={formatDuration(metrics.duration)}
            color="text-purple-600"
          />
        </div>
      </div>

      {/* Gantt Chart */}
      <div className="p-4 overflow-x-auto">
        <Gantt
          tasks={tasks}
          viewMode={ViewMode.Minute}
          columnWidth={65}
          listCellWidth="155px"
          barCornerRadius={4}
          barProgressColor="#3b82f6"
          barProgressSelectedColor="#2563eb"
          barBackgroundColor="#dbeafe"
          barBackgroundSelectedColor="#bfdbfe"
          todayColor="rgba(252, 165, 165, 0.3)"
          handleWidth={8}
          fontSize="12px"
          rowHeight={40}
          headerHeight={50}
        />
      </div>
    </div>
  );
}

function MetricCard({ icon, label, value, color }) {
  return (
    <div className="bg-white p-3 rounded-lg border">
      <div className="flex items-center gap-2 mb-1">
        <span className={color}>{icon}</span>
        <span className="text-sm text-gray-600">{label}</span>
      </div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

function getStepProgress(step) {
  switch (step.status) {
    case 'completed':
      return 100;
    case 'running':
      return 50;
    case 'failed':
    case 'error':
      return 100;
    case 'pending':
    default:
      return 0;
  }
}

function getTaskStyles(status) {
  switch (status) {
    case 'completed':
      return {
        backgroundColor: '#10b981',
        backgroundSelectedColor: '#059669',
        progressColor: '#047857',
        progressSelectedColor: '#065f46'
      };
    case 'failed':
    case 'error':
      return {
        backgroundColor: '#ef4444',
        backgroundSelectedColor: '#dc2626',
        progressColor: '#b91c1c',
        progressSelectedColor: '#991b1b'
      };
    case 'running':
      return {
        backgroundColor: '#f59e0b',
        backgroundSelectedColor: '#d97706',
        progressColor: '#b45309',
        progressSelectedColor: '#92400e'
      };
    case 'pending':
    default:
      return {
        backgroundColor: '#9ca3af',
        backgroundSelectedColor: '#6b7280',
        progressColor: '#4b5563',
        progressSelectedColor: '#374151'
      };
  }
}

function formatDuration(ms) {
  if (!ms || ms < 0) return '0s';
  
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

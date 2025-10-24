import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { Play, CheckCircle, XCircle, Loader, Circle } from 'lucide-react';

export default memo(({ data, isConnectable }) => {
  const status = data.status || 'pending';
  const { icon: Icon, color, bgColor } = getStatusStyle(status);

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 ${bgColor} ${color} shadow-md min-w-[150px] hover:shadow-lg transition-shadow`}
    >
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={isConnectable}
        className="w-3 h-3 !bg-blue-500"
      />

      <div className="flex items-center gap-2">
        <Icon className="w-4 h-4" />
        <div>
          <div className="font-medium text-sm">{data.label}</div>
          {data.stepType && <div className="text-xs opacity-75">{data.stepType}</div>}
        </div>
      </div>

      {data.duration && (
        <div className="text-xs mt-1 opacity-75">{formatDuration(data.duration)}</div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        isConnectable={isConnectable}
        className="w-3 h-3 !bg-blue-500"
      />
    </div>
  );
});

function getStatusStyle(status) {
  switch (status) {
    case 'completed':
      return {
        icon: CheckCircle,
        color: 'text-green-700',
        bgColor: 'bg-green-50 border-green-300',
      };
    case 'failed':
    case 'error':
      return {
        icon: XCircle,
        color: 'text-red-700',
        bgColor: 'bg-red-50 border-red-300',
      };
    case 'running':
      return {
        icon: Loader,
        color: 'text-yellow-700',
        bgColor: 'bg-yellow-50 border-yellow-300',
      };
    case 'pending':
    default:
      return {
        icon: Circle,
        color: 'text-gray-700',
        bgColor: 'bg-white border-gray-300',
      };
  }
}

function formatDuration(ms) {
  if (!ms) return '';
  const seconds = Math.floor(ms / 1000);
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

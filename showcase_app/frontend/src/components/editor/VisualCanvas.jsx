import { useState, useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Panel
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Plus, Save, Code, Eye, Columns } from 'lucide-react';
import StepNode from './StepNode';
import ParallelNode from './ParallelNode';
import DecisionNode from './DecisionNode';
import ModulePalette from './ModulePalette';

const nodeTypes = {
  step: StepNode,
  parallel: ParallelNode,
  decision: DecisionNode,
};

const edgeTypes = {
  default: 'smoothstep',
};

export default function VisualCanvas({ pipelineConfig, onConfigChange }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);

  // Initialize from pipeline config
  useState(() => {
    if (pipelineConfig) {
      const { nodes: initialNodes, edges: initialEdges } = convertConfigToGraph(pipelineConfig);
      setNodes(initialNodes);
      setEdges(initialEdges);
    }
  }, [pipelineConfig]);

  const onConnect = useCallback(
    (params) => {
      const newEdges = addEdge(
        {
          ...params,
          type: 'smoothstep',
          animated: false,
          style: { stroke: '#3b82f6' },
        },
        edges
      );
      setEdges(newEdges);
      updatePipelineConfig(nodes, newEdges);
    },
    [edges, nodes]
  );

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  const onNodesDelete = useCallback(
    (deleted) => {
      const remainingNodes = nodes.filter((n) => !deleted.find((d) => d.id === n.id));
      setNodes(remainingNodes);
      updatePipelineConfig(remainingNodes, edges);
    },
    [nodes, edges]
  );

  const onEdgesDelete = useCallback(
    (deleted) => {
      const remainingEdges = edges.filter((e) => !deleted.find((d) => d.id === e.id));
      setEdges(remainingEdges);
      updatePipelineConfig(nodes, remainingEdges);
    },
    [nodes, edges]
  );

  const addStepNode = useCallback(
    (stepType) => {
      const newNode = {
        id: `step-${Date.now()}`,
        type: 'step',
        position: {
          x: Math.random() * 400 + 100,
          y: Math.random() * 400 + 100,
        },
        data: {
          label: `New ${stepType}`,
          stepType: stepType,
          config: {},
        },
      };
      setNodes((nds) => [...nds, newNode]);
      updatePipelineConfig([...nodes, newNode], edges);
    },
    [nodes, edges]
  );

  const updateNodeData = useCallback(
    (nodeId, newData) => {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              data: { ...node.data, ...newData },
            };
          }
          return node;
        })
      );
      const updatedNodes = nodes.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...newData } } : node
      );
      updatePipelineConfig(updatedNodes, edges);
    },
    [nodes, edges]
  );

  const updatePipelineConfig = (currentNodes, currentEdges) => {
    const config = convertGraphToConfig(currentNodes, currentEdges);
    if (onConfigChange) {
      onConfigChange(config);
    }
  };

  return (
    <div className="h-full flex">
      {/* Module Palette */}
      <ModulePalette onAddStep={addStepNode} />

      {/* ReactFlow Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onNodesDelete={onNodesDelete}
          onEdgesDelete={onEdgesDelete}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          attributionPosition="bottom-left"
        >
          <Background color="#e5e7eb" gap={16} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              switch (node.type) {
                case 'step':
                  return '#3b82f6';
                case 'parallel':
                  return '#8b5cf6';
                case 'decision':
                  return '#f59e0b';
                default:
                  return '#6b7280';
              }
            }}
          />

          {/* Top Panel */}
          <Panel position="top-right">
            <div className="bg-white rounded-lg shadow-lg p-2 flex gap-2">
              <button
                className="px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-2"
                onClick={() => updatePipelineConfig(nodes, edges)}
              >
                <Save className="w-4 h-4" />
                Save
              </button>
            </div>
          </Panel>
        </ReactFlow>
      </div>

      {/* Property Panel */}
      {selectedNode && (
        <PropertyPanel
          node={selectedNode}
          onUpdate={(data) => updateNodeData(selectedNode.id, data)}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
}

function PropertyPanel({ node, onUpdate, onClose }) {
  const [label, setLabel] = useState(node.data.label || '');
  const [config, setConfig] = useState(JSON.stringify(node.data.config || {}, null, 2));

  const handleSave = () => {
    try {
      const parsedConfig = JSON.parse(config);
      onUpdate({ label, config: parsedConfig });
    } catch (e) {
      alert('Invalid JSON configuration');
    }
  };

  return (
    <div className="w-80 bg-white border-l shadow-lg p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Node Properties</h3>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
          ×
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
          <input
            type="text"
            value={node.data.stepType || node.type}
            disabled
            className="w-full px-3 py-2 border rounded-lg bg-gray-50"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Configuration</label>
          <textarea
            value={config}
            onChange={(e) => setConfig(e.target.value)}
            rows={10}
            className="w-full px-3 py-2 border rounded-lg font-mono text-xs focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <button
          onClick={handleSave}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Apply Changes
        </button>
      </div>
    </div>
  );
}

// Convert pipeline config to ReactFlow graph
function convertConfigToGraph(config) {
  const nodes = [];
  const edges = [];

  // Convert steps to nodes
  if (config.steps) {
    config.steps.forEach((step, index) => {
      nodes.push({
        id: step.id || step.name,
        type: 'step',
        position: { x: 100 + index * 200, y: 100 },
        data: {
          label: step.name,
          stepType: step.type || 'task',
          config: step.config || {},
        },
      });
    });
  }

  // Convert flow paths to edges
  if (config.flow?.paths) {
    config.flow.paths.forEach((path, index) => {
      edges.push({
        id: `edge-${index}`,
        source: path.from || path.from_step,
        target: path.to || path.to_step,
        type: 'smoothstep',
        animated: false,
        label: path.condition?.description || '',
      });
    });
  }

  return { nodes, edges };
}

// Convert ReactFlow graph to pipeline config
function convertGraphToConfig(nodes, edges) {
  const steps = nodes.map((node) => ({
    id: node.id,
    name: node.data.label,
    type: node.data.stepType || 'task',
    config: node.data.config || {},
  }));

  const paths = edges.map((edge) => ({
    from: edge.source,
    to: edge.target,
    condition: edge.label ? { description: edge.label } : { type: 'always' },
  }));

  return {
    name: 'Visual Pipeline',
    steps,
    flow: {
      start_at: steps[0]?.id || '',
      paths,
    },
  };
}

import { useState, useEffect, useCallback } from 'react';
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
import { Plus, Save, Code, Eye, Columns, FileCode } from 'lucide-react';
import StepNode from './StepNode';
import ParallelNode from './ParallelNode';
import DecisionNode from './DecisionNode';
import ModulePalette from './ModulePalette';
import StepCodeEditor from './StepCodeEditor';

const nodeTypes = {
  step: StepNode,
  parallel: ParallelNode,
  decision: DecisionNode,
};

const edgeTypes = {
  default: 'smoothstep',
};

export default function VisualCanvas({ pipelineConfig, pipelineId, onConfigChange }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const [selectedStepForCode, setSelectedStepForCode] = useState(null);

  // Initialize from pipeline config
  useEffect(() => {
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
    // Don't auto-open code editor on click, only when button is pressed
  }, []);

  const handleViewCode = useCallback((node) => {
    setSelectedStepForCode(node);
    setShowCodeEditor(true);
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
          onViewCode={() => handleViewCode(selectedNode)}
          pipelineId={pipelineId}
        />
      )}

      {/* Step Code Editor Sidebar */}
      {showCodeEditor && selectedStepForCode && pipelineId && (
        <div className="absolute top-0 right-0 w-1/2 h-full border-l bg-white shadow-2xl z-50">
          <StepCodeEditor
            pipelineId={pipelineId}
            stepId={selectedStepForCode.id}
            onClose={() => {
              setShowCodeEditor(false);
              setSelectedStepForCode(null);
            }}
          />
        </div>
      )}
    </div>
  );
}

function PropertyPanel({ node, onUpdate, onClose, onViewCode, pipelineId }) {
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

        <div className="space-y-2">
          <button
            onClick={handleSave}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Apply Changes
          </button>

          {/* View Code button - only show for step nodes and if pipeline has ID */}
          {node.type === 'step' && pipelineId && onViewCode && (
            <button
              onClick={onViewCode}
              className="w-full px-4 py-2 border border-blue-600 text-blue-600 rounded-lg hover:bg-blue-50 flex items-center justify-center gap-2"
            >
              <FileCode className="w-4 h-4" />
              View Step Code
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Convert pipeline config to ReactFlow graph
function convertConfigToGraph(config) {
  const nodes = [];
  const edges = [];

  if (!config.steps || config.steps.length === 0) {
    return { nodes, edges };
  }

  // Build adjacency lists for graph traversal
  const outgoing = {}; // from_step -> [to_steps]
  const incoming = {}; // to_step -> [from_steps]

  config.steps.forEach(step => {
    outgoing[step.id] = [];
    incoming[step.id] = [];
  });

  // Convert flow paths to edges and build adjacency lists
  if (config.flow?.paths) {
    config.flow.paths.forEach((path, index) => {
      edges.push({
        id: `edge-${index}`,
        source: path.from,
        target: path.to,
        type: 'smoothstep',
        animated: false,
        label: path.condition?.description || path.condition?.type,
      });

      if (outgoing[path.from]) {
        outgoing[path.from].push(path.to);
      }
      if (incoming[path.to]) {
        incoming[path.to].push(path.from);
      }
    });
  }

  // Layout algorithm: hierarchical layout for DAG
  const positions = {};
  const levels = {}; // step_id -> level (depth from start)
  const visited = new Set();

  // Find start node
  const startStep = config.flow?.start_at || config.steps[0].id;

  // BFS to assign levels
  const queue = [[startStep, 0]];
  visited.add(startStep);
  levels[startStep] = 0;
  let maxLevel = 0;

  while (queue.length > 0) {
    const [currentId, level] = queue.shift();
    maxLevel = Math.max(maxLevel, level);

    const children = outgoing[currentId] || [];
    children.forEach(childId => {
      if (!visited.has(childId)) {
        visited.add(childId);
        levels[childId] = level + 1;
        queue.push([childId, level + 1]);
      }
    });
  }

  // Steps without level (disconnected) get placed at the end
  config.steps.forEach(step => {
    if (levels[step.id] === undefined) {
      levels[step.id] = maxLevel + 1;
    }
  });

  // Group nodes by level
  const nodesByLevel = {};
  config.steps.forEach(step => {
    const level = levels[step.id];
    if (!nodesByLevel[level]) {
      nodesByLevel[level] = [];
    }
    nodesByLevel[level].push(step);
  });

  // Assign positions: spread out nodes at same level vertically
  const horizontalSpacing = 300;
  const verticalSpacing = 150;
  const startX = 100;
  const startY = 100;

  Object.keys(nodesByLevel).forEach(level => {
    const levelNodes = nodesByLevel[level];
    const levelInt = parseInt(level);

    // Calculate total height for this level
    const totalHeight = (levelNodes.length - 1) * verticalSpacing;
    const startYForLevel = startY - totalHeight / 2;

    levelNodes.forEach((step, index) => {
      positions[step.id] = {
        x: startX + levelInt * horizontalSpacing,
        y: startYForLevel + index * verticalSpacing
      };
    });
  });

  // Convert steps to nodes with calculated positions
  config.steps.forEach(step => {
    nodes.push({
      id: step.id,
      type: 'step',
      position: positions[step.id] || { x: 100, y: 100 },
      data: {
        label: step.name,
        stepType: step.type,
        config: step.config,
      },
    });
  });

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

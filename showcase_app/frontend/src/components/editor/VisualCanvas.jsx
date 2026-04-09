import { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Panel,
  getBezierPath,
  BaseEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';
import { Plus, Save, Code, Eye, Columns, FileCode } from 'lucide-react';
import StepNode from './StepNode';
import ParallelNode from './ParallelNode';
import DecisionNode from './DecisionNode';
import ModulePalette from './ModulePalette';
import StepCodeEditor from './StepCodeEditor';
import { formatEdge } from '../graph/utils/edgeFormatter';

// Custom edge that positions label near the source (20% along path)
function SourceLabelEdge({
  id, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, label, style, animated, markerEnd,
  data,
}) {
  const [edgePath] = getBezierPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
  });

  // Get point at ~20% along the bezier for label placement
  const t = 0.2;
  const labelX = (1 - t) * (1 - t) * (1 - t) * sourceX
    + 3 * (1 - t) * (1 - t) * t * ((sourceX + targetX) / 2)
    + 3 * (1 - t) * t * t * ((sourceX + targetX) / 2)
    + t * t * t * targetX;
  const labelY = (1 - t) * (1 - t) * (1 - t) * sourceY
    + 3 * (1 - t) * (1 - t) * t * sourceY
    + 3 * (1 - t) * t * t * targetY
    + t * t * t * targetY;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={style}
        markerEnd={markerEnd}
      />
      {label && (
        <foreignObject
          x={labelX - 4}
          y={labelY - 10}
          width={160}
          height={24}
          requiredExtensions="http://www.w3.org/1999/xhtml"
          className="overflow-visible pointer-events-none"
        >
          <div
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: '#6b7280',
              background: 'rgba(249,250,251,0.92)',
              padding: '2px 6px',
              borderRadius: 4,
              whiteSpace: 'nowrap',
              width: 'fit-content',
            }}
          >
            {label}
          </div>
        </foreignObject>
      )}
    </>
  );
}

const nodeTypes = {
  step: StepNode,
  parallel: ParallelNode,
  decision: DecisionNode,
};

const edgeTypes = {
  sourceLabel: SourceLabelEdge,
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
          type: 'sourceLabel',
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
    const config = convertGraphToConfig(currentNodes, currentEdges, pipelineConfig);
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
          defaultEdgeOptions={{
            type: 'sourceLabel',
            style: { strokeWidth: 1.5 },
          }}
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
            <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-2 flex gap-2">
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
        <div className="absolute top-0 right-0 w-1/2 h-full border-l dark:border-gray-700 bg-white dark:bg-gray-900 shadow-2xl z-50">
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
    <div className="w-80 bg-white dark:bg-gray-900 border-l dark:border-gray-700 shadow-lg p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">Node Properties</h3>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300">
          ×
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Name</label>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Type</label>
          <input
            type="text"
            value={node.data.stepType || node.type}
            disabled
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-400"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Configuration</label>
          <textarea
            value={config}
            onChange={(e) => setConfig(e.target.value)}
            rows={10}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 font-mono text-xs focus:ring-2 focus:ring-blue-500"
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
              className="w-full px-4 py-2 border border-blue-600 dark:border-blue-500 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-950/30 flex items-center justify-center gap-2"
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
  // Support both 'from_step'/'to_step' (pipeline JSON) and 'from'/'to' (legacy) formats
  // First pass: build adjacency lists
  const normalizedPaths = [];
  if (config.flow?.paths) {
    config.flow.paths.forEach((path) => {
      const fromStep = path.from_step || path.from;
      const toStep = path.to_step || path.to;
      normalizedPaths.push({ from_step: fromStep, to_step: toStep, condition: path.condition });

      if (outgoing[fromStep]) {
        outgoing[fromStep].push(toStep);
      }
      if (incoming[toStep]) {
        incoming[toStep].push(fromStep);
      }
    });
  }

  // BFS to assign levels (needed for back-edge detection)
  const levels = {};
  const visited = new Set();
  const startStep = config.flow?.start_at || config.steps[0].id;
  const queue = [[startStep, 0]];
  visited.add(startStep);
  levels[startStep] = 0;
  let maxLevel = 0;

  while (queue.length > 0) {
    const [currentId, level] = queue.shift();
    maxLevel = Math.max(maxLevel, level);
    (outgoing[currentId] || []).forEach(childId => {
      if (!visited.has(childId)) {
        visited.add(childId);
        levels[childId] = level + 1;
        queue.push([childId, level + 1]);
      }
    });
  }
  config.steps.forEach(step => {
    if (levels[step.id] === undefined) levels[step.id] = maxLevel + 1;
  });

  // Build step lookups needed for edge generation and port inference
  const stepsById = {};
  const stepNameMap = {};
  config.steps.forEach(step => {
    stepsById[step.id] = step;
    stepNameMap[step.id] = step.name;
  });

  // Build a lookup: for each step, index its inputs by source step ID
  // This lets us find which specific ports an edge should connect to
  const stepInputsBySource = {}; // targetStepId -> { sourceStepId -> [{ inputName, outputField }] }

  const sourcePattern = /^\{steps\.([^.]+)\.output\.([^}]+)\}$/;
  config.steps.forEach(step => {
    const inputs = step.inputs || [];
    const inputList = Array.isArray(inputs)
      ? inputs.filter(i => typeof i === 'object' && i.name)
      : Object.entries(inputs).map(([k, v]) => ({ name: k, source: v }));

    inputList.forEach(inp => {
      const match = sourcePattern.exec(inp.source || '');
      if (match) {
        const [, srcStepId, outputField] = match;
        if (!stepInputsBySource[step.id]) stepInputsBySource[step.id] = {};
        if (!stepInputsBySource[step.id][srcStepId]) stepInputsBySource[step.id][srcStepId] = [];
        stepInputsBySource[step.id][srcStepId].push({ inputName: inp.name, outputField });
      }
    });
  });

  // Second pass: generate edges with human-readable labels and back-edge detection
  normalizedPaths.forEach((path, index) => {
    const sourceLevel = levels[path.from_step] ?? 0;
    const targetLevel = levels[path.to_step] ?? 0;
    const isBackEdge = targetLevel <= sourceLevel && path.from_step !== path.to_step;

    const { label, style } = formatEdge(path, isBackEdge);

    // Determine port handles for this edge
    const fromStep = stepsById[path.from_step];
    const toStep = stepsById[path.to_step];
    const fromHasExplicitOutputs = fromStep && normalizePortNames(fromStep.outputs).length > 0;
    const toHasExplicitInputs = toStep && normalizePortNames(toStep.inputs).length > 0;

    let sourceHandle = undefined;
    let targetHandle = undefined;

    // Check if there are explicit source references linking specific ports
    const portLinks = stepInputsBySource[path.to_step]?.[path.from_step];

    if (portLinks && portLinks.length > 0) {
      // Use the first matching port link for this edge
      // (multiple links between same steps share one flow edge)
      sourceHandle = `out-${portLinks[0].outputField}`;
      targetHandle = `in-${portLinks[0].inputName}`;
    } else if (!fromHasExplicitOutputs && !toHasExplicitInputs) {
      // Inferred ports: handle IDs are based on connected step names
      const fromName = stepNameMap[path.from_step] || path.from_step;
      const toName = stepNameMap[path.to_step] || path.to_step;
      sourceHandle = `out-${toName}`;
      targetHandle = `in-${fromName}`;
    } else {
      // Mixed: one side has explicit ports, other doesn't — pick first available
      if (fromHasExplicitOutputs) {
        const firstOutput = normalizePortNames(fromStep.outputs)[0];
        if (firstOutput) sourceHandle = `out-${firstOutput}`;
      } else {
        const toName = stepNameMap[path.to_step] || path.to_step;
        sourceHandle = `out-${toName}`;
      }
      if (toHasExplicitInputs) {
        const firstInput = normalizePortNames(toStep.inputs)[0];
        if (firstInput) targetHandle = `in-${firstInput}`;
      } else {
        const fromName = stepNameMap[path.from_step] || path.from_step;
        targetHandle = `in-${fromName}`;
      }
    }

    edges.push({
      id: `edge-${index}`,
      source: path.from_step,
      target: path.to_step,
      sourceHandle,
      targetHandle,
      type: 'sourceLabel',
      animated: !isBackEdge,
      label: label || (path.condition?.type === 'always' ? undefined : path.condition?.type),
      style,
    });
  });

  // Build inferred ports from flow connections for steps that lack explicit inputs/outputs
  const inferredInputs = {};  // stepId -> Set of source step names
  const inferredOutputs = {}; // stepId -> Set of target step names

  normalizedPaths.forEach(path => {
    const fromId = path.from_step;
    const toId = path.to_step;
    if (!inferredOutputs[fromId]) inferredOutputs[fromId] = new Set();
    if (!inferredInputs[toId]) inferredInputs[toId] = new Set();
    inferredOutputs[fromId].add(stepNameMap[toId] || toId);
    inferredInputs[toId].add(stepNameMap[fromId] || fromId);
  });

  // Resolve final port lists per step (needed for accurate node sizing)
  const resolvedPorts = {};
  config.steps.forEach(step => {
    const explicitInputs = normalizePortNames(step.inputs);
    const explicitOutputs = normalizePortNames(step.outputs);
    resolvedPorts[step.id] = {
      inputs: explicitInputs.length > 0
        ? explicitInputs
        : [...(inferredInputs[step.id] || [])],
      outputs: explicitOutputs.length > 0
        ? explicitOutputs
        : [...(inferredOutputs[step.id] || [])],
    };
  });

  // Dagre layout — hierarchical with edge-crossing minimization
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: 'LR',       // left-to-right flow
    nodesep: 60,          // vertical gap between nodes in same rank
    ranksep: 200,         // horizontal gap between ranks
    edgesep: 30,          // gap between edges
    marginx: 40,
    marginy: 40,
    ranker: 'network-simplex', // best crossing minimization
  });
  g.setDefaultEdgeLabel(() => ({}));

  // Add nodes to dagre with estimated sizes based on actual port count
  const NODE_BASE_WIDTH = 200;
  const NODE_BASE_HEIGHT = 70;
  const PORT_ROW_HEIGHT = 22;

  config.steps.forEach(step => {
    const { inputs: ins, outputs: outs } = resolvedPorts[step.id];
    const portRows = Math.max(ins.length, outs.length, 0);
    const height = NODE_BASE_HEIGHT + portRows * PORT_ROW_HEIGHT;
    g.setNode(step.id, { width: NODE_BASE_WIDTH, height });
  });

  // Add forward edges to dagre (skip back-edges to avoid layout cycles)
  const dagreForwardTargets = new Set(); // nodes that have at least one forward incoming edge
  normalizedPaths.forEach(path => {
    const sourceLevel = levels[path.from_step] ?? 0;
    const targetLevel = levels[path.to_step] ?? 0;
    const isBackEdge = targetLevel <= sourceLevel && path.from_step !== path.to_step;
    if (!isBackEdge) {
      g.setEdge(path.from_step, path.to_step);
      dagreForwardTargets.add(path.to_step);
    }
  });

  // For nodes that lost all incoming edges (only reached via back-edges),
  // add a phantom edge from any node at the previous BFS level so dagre
  // places them at the correct rank instead of rank 0.
  const nodesByLevel = {};
  config.steps.forEach(step => {
    const lvl = levels[step.id] ?? 0;
    if (!nodesByLevel[lvl]) nodesByLevel[lvl] = [];
    nodesByLevel[lvl].push(step.id);
  });

  config.steps.forEach(step => {
    const lvl = levels[step.id] ?? 0;
    if (lvl > 0 && !dagreForwardTargets.has(step.id)) {
      // This node has no forward incoming edges — anchor it to correct rank
      const prevLevelNodes = nodesByLevel[lvl - 1];
      if (prevLevelNodes && prevLevelNodes.length > 0) {
        g.setEdge(prevLevelNodes[0], step.id, { weight: 0, minlen: 1 });
      }
    }
  });

  dagre.layout(g);

  // Extract positions from dagre (center coords → top-left for ReactFlow)
  const positions = {};
  config.steps.forEach(step => {
    const node = g.node(step.id);
    if (node) {
      positions[step.id] = {
        x: node.x - node.width / 2,
        y: node.y - node.height / 2,
      };
    }
  });

  // Convert steps to nodes with calculated positions
  config.steps.forEach(step => {
    const { inputs, outputs } = resolvedPorts[step.id];
    nodes.push({
      id: step.id,
      type: 'step',
      position: positions[step.id] || { x: 100, y: 100 },
      data: {
        label: step.name,
        stepType: step.type,
        config: step.config,
        inputs,
        outputs,
      },
    });
  });

  return { nodes, edges };
}

// Convert ReactFlow graph to pipeline config
function convertGraphToConfig(nodes, edges, existingConfig) {
  const steps = nodes.map((node) => ({
    id: node.id,
    name: node.data.label,
    type: node.data.stepType || 'task',
    config: node.data.config || {},
  }));

  const paths = edges.map((edge) => ({
    from_step: edge.source,
    to_step: edge.target,
    condition: edge.label ? { description: edge.label } : { type: 'always' },
  }));

  return {
    ...existingConfig,
    steps,
    flow: {
      start_at: steps[0]?.id || '',
      paths,
    },
  };
}

/**
 * Normalize various input/output formats into a simple array of port name strings.
 * Handles: array of objects [{name: "x"}], object {x: "..."}, array of strings ["x"], or undefined.
 */
function normalizePortNames(ports) {
  if (!ports) return [];
  if (Array.isArray(ports)) {
    return ports.map(p => (typeof p === 'string' ? p : p.name)).filter(Boolean);
  }
  if (typeof ports === 'object') {
    return Object.keys(ports);
  }
  return [];
}

# Developer Experience Implementation Plan

**Date**: 2025-10-25
**Status**: Planning Phase
**Priority**: Medium - Developer Productivity

---

## Table of Contents

1. [Visual Pipeline Designer](#1-visual-pipeline-designer)
2. [Pipeline Debugger](#2-pipeline-debugger)
3. [Mock Data Generator](#3-mock-data-generator)
4. [Advanced CLI Tools](#4-advanced-cli-tools)
5. [IDE Integration](#5-ide-integration)
6. [Implementation Timeline](#implementation-timeline)
7. [Dependencies & Prerequisites](#dependencies--prerequisites)

---

## 1. Visual Pipeline Designer

### Overview
Create a web-based visual pipeline designer using React Flow that allows developers to design, validate, and export pipelines through an intuitive drag-and-drop interface.

### Requirements

#### 1.1 React Flow Designer Component

```typescript
// ia_modules/web/components/PipelineDesigner/PipelineDesigner.tsx

import React, { useCallback, useState, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  addEdge,
  Connection,
  EdgeChange,
  NodeChange,
  applyNodeChanges,
  applyEdgeChanges,
  Panel,
  useReactFlow,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { StepLibrary } from './StepLibrary';
import { PropertiesPanel } from './PropertiesPanel';
import { ValidationPanel } from './ValidationPanel';
import { ExportDialog } from './ExportDialog';
import { CustomNode } from './CustomNode';
import { CustomEdge } from './CustomEdge';

interface PipelineDesignerProps {
  initialPipeline?: PipelineDefinition;
  onSave?: (pipeline: PipelineDefinition) => void;
  readOnly?: boolean;
}

interface PipelineDefinition {
  id?: string;
  name: string;
  description?: string;
  steps: Step[];
  version?: string;
}

interface Step {
  id: string;
  name: string;
  type: string;
  module: string;
  inputs?: Record<string, any>;
  outputs?: Record<string, any>;
  position?: { x: number; y: number };
  nextSteps?: string[];
}

const nodeTypes = {
  stepNode: CustomNode,
};

const edgeTypes = {
  conditional: CustomEdge,
};

export const PipelineDesigner: React.FC<PipelineDesignerProps> = ({
  initialPipeline,
  onSave,
  readOnly = false,
}) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const reactFlowInstance = useReactFlow();

  // Load initial pipeline
  React.useEffect(() => {
    if (initialPipeline) {
      loadPipeline(initialPipeline);
    }
  }, [initialPipeline]);

  const loadPipeline = (pipeline: PipelineDefinition) => {
    const loadedNodes: Node[] = pipeline.steps.map((step, index) => ({
      id: step.id,
      type: 'stepNode',
      position: step.position || { x: 250 * index, y: 100 },
      data: {
        label: step.name,
        stepType: step.type,
        module: step.module,
        inputs: step.inputs,
        outputs: step.outputs,
      },
    }));

    const loadedEdges: Edge[] = [];
    pipeline.steps.forEach(step => {
      if (step.nextSteps) {
        step.nextSteps.forEach(nextStepId => {
          loadedEdges.push({
            id: `${step.id}-${nextStepId}`,
            source: step.id,
            target: nextStepId,
            type: 'conditional',
          });
        });
      }
    });

    setNodes(loadedNodes);
    setEdges(loadedEdges);
  };

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (!readOnly) {
        setNodes((nds) => applyNodeChanges(changes, nds));
      }
    },
    [readOnly]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (!readOnly) {
        setEdges((eds) => applyEdgeChanges(changes, eds));
      }
    },
    [readOnly]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!readOnly) {
        const newEdge = {
          ...connection,
          type: 'conditional',
          id: `${connection.source}-${connection.target}`,
        } as Edge;
        setEdges((eds) => addEdge(newEdge, eds));
      }
    },
    [readOnly]
  );

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      if (typeof type === 'undefined' || !type || readOnly) {
        return;
      }

      const stepData = JSON.parse(type);
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: Node = {
        id: `step-${Date.now()}`,
        type: 'stepNode',
        position,
        data: {
          label: stepData.name,
          stepType: stepData.type,
          module: stepData.module,
          inputs: {},
          outputs: {},
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, readOnly]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const validatePipeline = useCallback(() => {
    const errors: string[] = [];

    // Check for cycles
    if (hasCycles(nodes, edges)) {
      errors.push('Pipeline contains cycles');
    }

    // Check for disconnected nodes
    const disconnectedNodes = findDisconnectedNodes(nodes, edges);
    if (disconnectedNodes.length > 0) {
      errors.push(
        `Disconnected nodes: ${disconnectedNodes.map(n => n.data.label).join(', ')}`
      );
    }

    // Check for missing required inputs
    nodes.forEach(node => {
      const missingInputs = checkMissingInputs(node, edges);
      if (missingInputs.length > 0) {
        errors.push(
          `Node "${node.data.label}" missing inputs: ${missingInputs.join(', ')}`
        );
      }
    });

    setValidationErrors(errors);
    return errors.length === 0;
  }, [nodes, edges]);

  const exportPipeline = useCallback(() => {
    if (!validatePipeline()) {
      alert('Pipeline has validation errors. Please fix them before exporting.');
      return;
    }

    const pipeline: PipelineDefinition = {
      name: 'Generated Pipeline',
      description: 'Created with Visual Pipeline Designer',
      steps: nodes.map(node => ({
        id: node.id,
        name: node.data.label,
        type: node.data.stepType,
        module: node.data.module,
        inputs: node.data.inputs,
        outputs: node.data.outputs,
        position: node.position,
        nextSteps: edges
          .filter(edge => edge.source === node.id)
          .map(edge => edge.target),
      })),
      version: '1.0',
    };

    setShowExportDialog(true);
    if (onSave) {
      onSave(pipeline);
    }
  }, [nodes, edges, validatePipeline, onSave]);

  const autoLayout = useCallback(() => {
    // Dagre layout algorithm
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    dagreGraph.setGraph({ rankdir: 'LR', nodesep: 100, ranksep: 200 });

    nodes.forEach(node => {
      dagreGraph.setNode(node.id, { width: 200, height: 80 });
    });

    edges.forEach(edge => {
      dagreGraph.setEdge(edge.source, edge.target);
    });

    dagre.layout(dagreGraph);

    const layoutedNodes = nodes.map(node => {
      const nodeWithPosition = dagreGraph.node(node.id);
      return {
        ...node,
        position: {
          x: nodeWithPosition.x - 100,
          y: nodeWithPosition.y - 40,
        },
      };
    });

    setNodes(layoutedNodes);
  }, [nodes, edges]);

  const updateNodeData = useCallback((nodeId: string, data: any) => {
    setNodes(nds =>
      nds.map(node =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node
      )
    );
  }, []);

  return (
    <div className="pipeline-designer" style={{ width: '100%', height: '100vh' }}>
      <div className="designer-layout" style={{ display: 'flex', height: '100%' }}>
        {/* Step Library Sidebar */}
        {!readOnly && (
          <div className="step-library-panel" style={{ width: '300px', borderRight: '1px solid #ccc' }}>
            <StepLibrary />
          </div>
        )}

        {/* Main Designer Canvas */}
        <div className="designer-canvas" style={{ flex: 1 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            attributionPosition="bottom-right"
          >
            <Background />
            <Controls />
            <MiniMap />
            <Panel position="top-left">
              <div className="toolbar" style={{ background: 'white', padding: '10px', borderRadius: '5px' }}>
                <button onClick={validatePipeline} disabled={readOnly}>
                  Validate
                </button>
                <button onClick={autoLayout} disabled={readOnly}>
                  Auto Layout
                </button>
                <button onClick={exportPipeline} disabled={readOnly}>
                  Export JSON
                </button>
              </div>
            </Panel>
          </ReactFlow>
        </div>

        {/* Properties Panel */}
        {selectedNode && (
          <div className="properties-panel" style={{ width: '350px', borderLeft: '1px solid #ccc' }}>
            <PropertiesPanel
              node={selectedNode}
              onUpdate={(data) => updateNodeData(selectedNode.id, data)}
              readOnly={readOnly}
            />
          </div>
        )}
      </div>

      {/* Validation Panel */}
      {validationErrors.length > 0 && (
        <ValidationPanel errors={validationErrors} />
      )}

      {/* Export Dialog */}
      {showExportDialog && (
        <ExportDialog
          pipeline={exportPipeline}
          onClose={() => setShowExportDialog(false)}
        />
      )}
    </div>
  );
};

// Helper functions
function hasCycles(nodes: Node[], edges: Edge[]): boolean {
  const graph = new Map<string, string[]>();
  edges.forEach(edge => {
    if (!graph.has(edge.source)) {
      graph.set(edge.source, []);
    }
    graph.get(edge.source)!.push(edge.target);
  });

  const visited = new Set<string>();
  const recStack = new Set<string>();

  function isCyclicUtil(nodeId: string): boolean {
    visited.add(nodeId);
    recStack.add(nodeId);

    const neighbors = graph.get(nodeId) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor) && isCyclicUtil(neighbor)) {
        return true;
      } else if (recStack.has(neighbor)) {
        return true;
      }
    }

    recStack.delete(nodeId);
    return false;
  }

  for (const node of nodes) {
    if (!visited.has(node.id) && isCyclicUtil(node.id)) {
      return true;
    }
  }

  return false;
}

function findDisconnectedNodes(nodes: Node[], edges: Edge[]): Node[] {
  if (nodes.length <= 1) return [];

  const connectedNodes = new Set<string>();
  edges.forEach(edge => {
    connectedNodes.add(edge.source);
    connectedNodes.add(edge.target);
  });

  return nodes.filter(node => !connectedNodes.has(node.id));
}

function checkMissingInputs(node: Node, edges: Edge[]): string[] {
  // Implement based on step schema validation
  return [];
}

// Wrapper component with ReactFlowProvider
export const PipelineDesignerWrapper: React.FC<PipelineDesignerProps> = (props) => (
  <ReactFlowProvider>
    <PipelineDesigner {...props} />
  </ReactFlowProvider>
);
```

#### 1.2 Custom Node Component

```typescript
// ia_modules/web/components/PipelineDesigner/CustomNode.tsx

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { FiPlay, FiSettings, FiCheck, FiAlertCircle } from 'react-icons/fi';

const CustomNode: React.FC<NodeProps> = ({ data, selected }) => {
  const getStatusIcon = () => {
    switch (data.status) {
      case 'running':
        return <FiPlay className="status-icon running" />;
      case 'success':
        return <FiCheck className="status-icon success" />;
      case 'error':
        return <FiAlertCircle className="status-icon error" />;
      default:
        return <FiSettings className="status-icon" />;
    }
  };

  const getNodeColor = () => {
    switch (data.stepType) {
      case 'llm':
        return '#4CAF50';
      case 'transform':
        return '#2196F3';
      case 'condition':
        return '#FF9800';
      case 'integration':
        return '#9C27B0';
      default:
        return '#757575';
    }
  };

  return (
    <div
      className={`custom-node ${selected ? 'selected' : ''}`}
      style={{
        border: `2px solid ${getNodeColor()}`,
        borderRadius: '8px',
        padding: '15px',
        background: 'white',
        minWidth: '200px',
        boxShadow: selected ? '0 0 0 2px #1976d2' : '0 2px 4px rgba(0,0,0,0.1)',
      }}
    >
      <Handle type="target" position={Position.Top} />

      <div className="node-header" style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
        {getStatusIcon()}
        <div style={{ marginLeft: '8px', flex: 1 }}>
          <div className="node-title" style={{ fontWeight: 'bold', fontSize: '14px' }}>
            {data.label}
          </div>
          <div className="node-type" style={{ fontSize: '12px', color: '#666' }}>
            {data.stepType}
          </div>
        </div>
      </div>

      {data.inputs && Object.keys(data.inputs).length > 0 && (
        <div className="node-inputs" style={{ fontSize: '11px', color: '#999', marginTop: '8px' }}>
          Inputs: {Object.keys(data.inputs).length}
        </div>
      )}

      {data.outputs && Object.keys(data.outputs).length > 0 && (
        <div className="node-outputs" style={{ fontSize: '11px', color: '#999' }}>
          Outputs: {Object.keys(data.outputs).length}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export default memo(CustomNode);
```

#### 1.3 Step Library Component

```typescript
// ia_modules/web/components/PipelineDesigner/StepLibrary.tsx

import React, { useState } from 'react';
import { FiSearch } from 'react-icons/fi';

interface StepTemplate {
  id: string;
  name: string;
  type: string;
  category: string;
  module: string;
  description: string;
  icon?: string;
  requiredInputs?: string[];
  outputs?: string[];
}

const STEP_TEMPLATES: StepTemplate[] = [
  {
    id: 'llm-prompt',
    name: 'LLM Prompt',
    type: 'llm',
    category: 'AI',
    module: 'ia_modules.pipeline.steps.llm_step',
    description: 'Execute LLM prompt with configurable model',
    requiredInputs: ['prompt', 'model'],
    outputs: ['response', 'tokens_used'],
  },
  {
    id: 'data-transform',
    name: 'Data Transform',
    type: 'transform',
    category: 'Data',
    module: 'ia_modules.pipeline.steps.transform_step',
    description: 'Transform data using Python function',
    requiredInputs: ['data', 'transform_fn'],
    outputs: ['transformed_data'],
  },
  {
    id: 'conditional',
    name: 'Conditional Branch',
    type: 'condition',
    category: 'Control Flow',
    module: 'ia_modules.pipeline.steps.condition_step',
    description: 'Branch execution based on condition',
    requiredInputs: ['condition', 'true_path', 'false_path'],
    outputs: ['result'],
  },
  {
    id: 'api-call',
    name: 'API Call',
    type: 'integration',
    category: 'Integration',
    module: 'ia_modules.pipeline.steps.api_step',
    description: 'Make HTTP API request',
    requiredInputs: ['url', 'method'],
    outputs: ['response', 'status_code'],
  },
  {
    id: 'database-query',
    name: 'Database Query',
    type: 'integration',
    category: 'Data',
    module: 'ia_modules.pipeline.steps.db_step',
    description: 'Execute database query',
    requiredInputs: ['query', 'connection'],
    outputs: ['results'],
  },
  {
    id: 'loop',
    name: 'Loop',
    type: 'loop',
    category: 'Control Flow',
    module: 'ia_modules.pipeline.steps.loop_step',
    description: 'Iterate over collection',
    requiredInputs: ['items', 'loop_step'],
    outputs: ['results'],
  },
];

export const StepLibrary: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', ...new Set(STEP_TEMPLATES.map(s => s.category))];

  const filteredSteps = STEP_TEMPLATES.filter(step => {
    const matchesSearch =
      step.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      step.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory =
      selectedCategory === 'All' || step.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const onDragStart = (event: React.DragEvent, step: StepTemplate) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(step));
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="step-library" style={{ padding: '15px', height: '100%', overflow: 'auto' }}>
      <h3 style={{ marginTop: 0 }}>Step Library</h3>

      {/* Search */}
      <div className="search-box" style={{ position: 'relative', marginBottom: '15px' }}>
        <FiSearch
          style={{
            position: 'absolute',
            left: '10px',
            top: '50%',
            transform: 'translateY(-50%)',
            color: '#999',
          }}
        />
        <input
          type="text"
          placeholder="Search steps..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            width: '100%',
            padding: '8px 8px 8px 35px',
            border: '1px solid #ddd',
            borderRadius: '4px',
          }}
        />
      </div>

      {/* Category Filter */}
      <div className="category-filter" style={{ marginBottom: '15px' }}>
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          style={{
            width: '100%',
            padding: '8px',
            border: '1px solid #ddd',
            borderRadius: '4px',
          }}
        >
          {categories.map(cat => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>
      </div>

      {/* Step Templates */}
      <div className="step-templates">
        {filteredSteps.map(step => (
          <div
            key={step.id}
            className="step-template"
            draggable
            onDragStart={(e) => onDragStart(e, step)}
            style={{
              padding: '12px',
              marginBottom: '10px',
              border: '1px solid #ddd',
              borderRadius: '6px',
              cursor: 'grab',
              background: 'white',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
              e.currentTarget.style.borderColor = '#1976d2';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.boxShadow = 'none';
              e.currentTarget.style.borderColor = '#ddd';
            }}
          >
            <div
              className="step-name"
              style={{ fontWeight: 'bold', fontSize: '13px', marginBottom: '4px' }}
            >
              {step.name}
            </div>
            <div className="step-type" style={{ fontSize: '11px', color: '#666', marginBottom: '8px' }}>
              {step.category} • {step.type}
            </div>
            <div className="step-description" style={{ fontSize: '12px', color: '#999' }}>
              {step.description}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

#### 1.4 Properties Panel Component

```typescript
// ia_modules/web/components/PipelineDesigner/PropertiesPanel.tsx

import React, { useState, useEffect } from 'react';
import { Node } from 'reactflow';
import { FiX, FiSave } from 'react-icons/fi';

interface PropertiesPanelProps {
  node: Node;
  onUpdate: (data: any) => void;
  readOnly?: boolean;
}

export const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  node,
  onUpdate,
  readOnly = false,
}) => {
  const [localData, setLocalData] = useState(node.data);

  useEffect(() => {
    setLocalData(node.data);
  }, [node]);

  const handleInputChange = (field: string, value: any) => {
    setLocalData({
      ...localData,
      inputs: {
        ...localData.inputs,
        [field]: value,
      },
    });
  };

  const handleSave = () => {
    onUpdate(localData);
  };

  return (
    <div className="properties-panel" style={{ padding: '20px', height: '100%', overflow: 'auto' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
        <h3 style={{ margin: 0 }}>Properties</h3>
        {!readOnly && (
          <button onClick={handleSave} style={{ padding: '5px 10px' }}>
            <FiSave /> Save
          </button>
        )}
      </div>

      {/* Basic Info */}
      <div className="section" style={{ marginBottom: '20px' }}>
        <h4>Basic Information</h4>
        <div className="form-group" style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '13px' }}>
            Name
          </label>
          <input
            type="text"
            value={localData.label}
            onChange={(e) => setLocalData({ ...localData, label: e.target.value })}
            disabled={readOnly}
            style={{
              width: '100%',
              padding: '8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
            }}
          />
        </div>

        <div className="form-group" style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '13px' }}>
            Type
          </label>
          <input
            type="text"
            value={localData.stepType}
            disabled
            style={{
              width: '100%',
              padding: '8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              background: '#f5f5f5',
            }}
          />
        </div>

        <div className="form-group" style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '13px' }}>
            Module
          </label>
          <input
            type="text"
            value={localData.module}
            disabled
            style={{
              width: '100%',
              padding: '8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              background: '#f5f5f5',
              fontSize: '12px',
            }}
          />
        </div>
      </div>

      {/* Inputs */}
      <div className="section" style={{ marginBottom: '20px' }}>
        <h4>Inputs</h4>
        {localData.requiredInputs?.map((inputName: string) => (
          <div key={inputName} className="form-group" style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '13px' }}>
              {inputName}
              <span style={{ color: 'red' }}>*</span>
            </label>
            <input
              type="text"
              value={localData.inputs?.[inputName] || ''}
              onChange={(e) => handleInputChange(inputName, e.target.value)}
              disabled={readOnly}
              placeholder={`Enter ${inputName}`}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
              }}
            />
          </div>
        ))}

        {/* Add Custom Input */}
        {!readOnly && (
          <button
            onClick={() => {
              const newInput = prompt('Enter input name:');
              if (newInput) {
                handleInputChange(newInput, '');
              }
            }}
            style={{
              padding: '8px 12px',
              background: '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            + Add Input
          </button>
        )}
      </div>

      {/* Outputs */}
      <div className="section">
        <h4>Outputs</h4>
        {localData.outputs && Object.keys(localData.outputs).length > 0 ? (
          Object.keys(localData.outputs).map(outputName => (
            <div key={outputName} style={{ marginBottom: '10px', padding: '8px', background: '#f5f5f5', borderRadius: '4px' }}>
              <span style={{ fontWeight: 'bold', fontSize: '13px' }}>{outputName}</span>
            </div>
          ))
        ) : (
          <p style={{ color: '#999', fontSize: '13px' }}>No outputs defined</p>
        )}
      </div>
    </div>
  );
};
```

#### 1.5 Backend API for Designer

```python
# ia_modules/api/routes/pipeline_designer.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from pydantic import BaseModel
import json

from ia_modules.auth.middleware import get_current_user
from ia_modules.auth.models import User
from ia_modules.database import get_db
from ia_modules.pipeline.core import PipelineDefinition
from ia_modules.pipeline.validation import PipelineValidator

router = APIRouter(prefix="/api/designer", tags=["designer"])

class PipelineDesignRequest(BaseModel):
    name: str
    description: str | None = None
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ValidationRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

@router.post("/validate")
async def validate_pipeline_design(
    request: ValidationRequest,
    current_user: User = Depends(get_current_user),
):
    """Validate pipeline design for errors"""

    validator = PipelineValidator()

    # Convert nodes and edges to pipeline definition
    pipeline_def = convert_design_to_pipeline(request.nodes, request.edges)

    # Run validation
    validation_result = validator.validate(pipeline_def)

    return {
        "valid": validation_result.is_valid,
        "errors": validation_result.errors,
        "warnings": validation_result.warnings,
    }

@router.post("/export")
async def export_pipeline_design(
    request: PipelineDesignRequest,
    format: str = "json",
    current_user: User = Depends(get_current_user),
):
    """Export pipeline design to JSON"""

    # Convert to pipeline definition
    pipeline_def = convert_design_to_pipeline(request.nodes, request.edges)

    if format == "json":
        return {
            "pipeline": pipeline_def.dict(),
            "format": "json"
        }
    elif format == "yaml":
        import yaml
        return {
            "pipeline": yaml.dump(pipeline_def.dict()),
            "format": "yaml"
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

@router.post("/save")
async def save_pipeline_design(
    request: PipelineDesignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Save pipeline design to database"""

    pipeline_def = convert_design_to_pipeline(request.nodes, request.edges)

    # Save to database
    from ia_modules.pipeline.services import PipelineService
    pipeline_service = PipelineService(db)

    saved_pipeline = await pipeline_service.create_pipeline(
        name=request.name,
        description=request.description,
        definition=pipeline_def.dict(),
        user_id=current_user.id
    )

    return {
        "id": saved_pipeline.id,
        "name": saved_pipeline.name,
        "message": "Pipeline saved successfully"
    }

@router.get("/templates")
async def list_pipeline_templates(
    current_user: User = Depends(get_current_user),
):
    """List available pipeline templates"""

    templates = [
        {
            "id": "simple-llm",
            "name": "Simple LLM Pipeline",
            "description": "Basic pipeline with single LLM step",
            "category": "Basic",
            "nodes": [
                {
                    "id": "step-1",
                    "type": "stepNode",
                    "position": {"x": 250, "y": 100},
                    "data": {
                        "label": "LLM Prompt",
                        "stepType": "llm",
                        "module": "ia_modules.pipeline.steps.llm_step",
                    }
                }
            ],
            "edges": []
        },
        {
            "id": "conditional-pipeline",
            "name": "Conditional Pipeline",
            "description": "Pipeline with conditional branching",
            "category": "Advanced",
            "nodes": [
                {
                    "id": "step-1",
                    "type": "stepNode",
                    "position": {"x": 250, "y": 100},
                    "data": {
                        "label": "Input Processing",
                        "stepType": "transform",
                    }
                },
                {
                    "id": "step-2",
                    "type": "stepNode",
                    "position": {"x": 250, "y": 250},
                    "data": {
                        "label": "Condition Check",
                        "stepType": "condition",
                    }
                },
                {
                    "id": "step-3a",
                    "type": "stepNode",
                    "position": {"x": 100, "y": 400},
                    "data": {
                        "label": "Path A",
                        "stepType": "llm",
                    }
                },
                {
                    "id": "step-3b",
                    "type": "stepNode",
                    "position": {"x": 400, "y": 400},
                    "data": {
                        "label": "Path B",
                        "stepType": "llm",
                    }
                },
            ],
            "edges": [
                {"id": "e1-2", "source": "step-1", "target": "step-2"},
                {"id": "e2-3a", "source": "step-2", "target": "step-3a"},
                {"id": "e2-3b", "source": "step-2", "target": "step-3b"},
            ]
        },
    ]

    return {"templates": templates}

def convert_design_to_pipeline(nodes: List[Dict], edges: List[Dict]) -> PipelineDefinition:
    """Convert visual design to pipeline definition"""

    steps = []

    for node in nodes:
        step = {
            "id": node["id"],
            "name": node["data"]["label"],
            "type": node["data"]["stepType"],
            "module": node["data"]["module"],
            "inputs": node["data"].get("inputs", {}),
            "outputs": node["data"].get("outputs", {}),
            "next_steps": []
        }

        # Find next steps from edges
        for edge in edges:
            if edge["source"] == node["id"]:
                step["next_steps"].append(edge["target"])

        steps.append(step)

    return PipelineDefinition(steps=steps)
```

---

## 2. Pipeline Debugger

### Overview
Interactive debugging tools for step-through execution, breakpoints, and variable inspection.

### Requirements

#### 2.1 Debugger Core

```python
# ia_modules/pipeline/debugger.py

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class DebuggerCommand(str, Enum):
    """Debugger commands"""
    CONTINUE = "continue"
    STEP_OVER = "step_over"
    STEP_INTO = "step_into"
    STEP_OUT = "step_out"
    PAUSE = "pause"
    STOP = "stop"
    SET_BREAKPOINT = "set_breakpoint"
    REMOVE_BREAKPOINT = "remove_breakpoint"
    EVALUATE = "evaluate"

class DebuggerState(str, Enum):
    """Debugger execution states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class Breakpoint:
    """Breakpoint definition"""
    step_id: str
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0

@dataclass
class DebugFrame:
    """Execution frame for debugging"""
    step_id: str
    step_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    local_vars: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

@dataclass
class DebugSession:
    """Debug session state"""
    session_id: str
    pipeline_id: str
    state: DebuggerState = DebuggerState.IDLE
    current_step: Optional[str] = None
    call_stack: List[DebugFrame] = field(default_factory=list)
    breakpoints: Dict[str, Breakpoint] = field(default_factory=dict)
    watch_expressions: List[str] = field(default_factory=list)
    execution_history: List[DebugFrame] = field(default_factory=list)

class PipelineDebugger:
    """Interactive pipeline debugger"""

    def __init__(self):
        self.sessions: Dict[str, DebugSession] = {}
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.event_callbacks: List[Callable] = []

    def create_session(self, session_id: str, pipeline_id: str) -> DebugSession:
        """Create new debug session"""
        session = DebugSession(
            session_id=session_id,
            pipeline_id=pipeline_id
        )
        self.sessions[session_id] = session

        logger.info(f"Created debug session: {session_id}")
        return session

    def set_breakpoint(
        self,
        session_id: str,
        step_id: str,
        condition: Optional[str] = None
    ):
        """Set breakpoint at step"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        breakpoint = Breakpoint(
            step_id=step_id,
            condition=condition
        )
        session.breakpoints[step_id] = breakpoint

        logger.info(f"Breakpoint set at {step_id}")
        self._emit_event("breakpoint_set", {"step_id": step_id})

    def remove_breakpoint(self, session_id: str, step_id: str):
        """Remove breakpoint"""
        session = self.sessions.get(session_id)
        if session and step_id in session.breakpoints:
            del session.breakpoints[step_id]
            logger.info(f"Breakpoint removed from {step_id}")
            self._emit_event("breakpoint_removed", {"step_id": step_id})

    def add_watch(self, session_id: str, expression: str):
        """Add watch expression"""
        session = self.sessions.get(session_id)
        if session:
            session.watch_expressions.append(expression)
            logger.info(f"Added watch: {expression}")

    async def should_break(
        self,
        session_id: str,
        step_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if should break at step"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check if breakpoint exists
        breakpoint = session.breakpoints.get(step_id)
        if not breakpoint or not breakpoint.enabled:
            return False

        # Increment hit count
        breakpoint.hit_count += 1

        # Evaluate condition if exists
        if breakpoint.condition:
            try:
                result = eval(breakpoint.condition, {}, context)
                return bool(result)
            except Exception as e:
                logger.error(f"Breakpoint condition error: {e}")
                return True

        return True

    async def pause_execution(
        self,
        session_id: str,
        step_id: str,
        step_name: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Pause execution at step"""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Create debug frame
        frame = DebugFrame(
            step_id=step_id,
            step_name=step_name,
            inputs=inputs,
            local_vars=context.copy(),
            timestamp=asyncio.get_event_loop().time()
        )

        # Update session state
        session.state = DebuggerState.PAUSED
        session.current_step = step_id
        session.call_stack.append(frame)

        # Emit pause event
        self._emit_event("execution_paused", {
            "step_id": step_id,
            "step_name": step_name,
            "frame": frame
        })

        # Wait for command
        command = await self.command_queue.get()

        # Process command
        await self._process_command(session_id, command)

    async def _process_command(self, session_id: str, command: DebuggerCommand):
        """Process debugger command"""
        session = self.sessions.get(session_id)
        if not session:
            return

        if command == DebuggerCommand.CONTINUE:
            session.state = DebuggerState.RUNNING
            logger.info("Continuing execution")

        elif command == DebuggerCommand.STEP_OVER:
            session.state = DebuggerState.RUNNING
            # Will pause at next step
            logger.info("Stepping over")

        elif command == DebuggerCommand.STOP:
            session.state = DebuggerState.STOPPED
            logger.info("Stopping execution")

        self._emit_event("command_processed", {"command": command})

    def send_command(self, session_id: str, command: DebuggerCommand):
        """Send command to debugger"""
        self.command_queue.put_nowait(command)
        logger.info(f"Command sent: {command}")

    def inspect_variable(
        self,
        session_id: str,
        var_name: str
    ) -> Optional[Any]:
        """Inspect variable value"""
        session = self.sessions.get(session_id)
        if not session or not session.call_stack:
            return None

        current_frame = session.call_stack[-1]

        # Check in inputs
        if var_name in current_frame.inputs:
            return current_frame.inputs[var_name]

        # Check in outputs
        if var_name in current_frame.outputs:
            return current_frame.outputs[var_name]

        # Check in local vars
        if var_name in current_frame.local_vars:
            return current_frame.local_vars[var_name]

        return None

    def get_call_stack(self, session_id: str) -> List[DebugFrame]:
        """Get current call stack"""
        session = self.sessions.get(session_id)
        if session:
            return session.call_stack.copy()
        return []

    def get_execution_history(self, session_id: str) -> List[DebugFrame]:
        """Get execution history for time-travel debugging"""
        session = self.sessions.get(session_id)
        if session:
            return session.execution_history.copy()
        return []

    def register_event_callback(self, callback: Callable):
        """Register callback for debug events"""
        self.event_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit debug event"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }

        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

# Global debugger instance
_debugger: Optional[PipelineDebugger] = None

def get_debugger() -> PipelineDebugger:
    """Get global debugger instance"""
    global _debugger
    if _debugger is None:
        _debugger = PipelineDebugger()
    return _debugger
```

#### 2.2 Debug-Enabled Pipeline Runner

```python
# ia_modules/pipeline/debug_runner.py

from typing import Dict, Any, Optional
import logging

from ia_modules.pipeline.runner import PipelineRunner
from ia_modules.pipeline.debugger import get_debugger, DebuggerState
from ia_modules.pipeline.core import PipelineContext

logger = logging.getLogger(__name__)

class DebugPipelineRunner(PipelineRunner):
    """Pipeline runner with debugging support"""

    def __init__(
        self,
        pipeline_definition: dict,
        debug_session_id: Optional[str] = None
    ):
        super().__init__(pipeline_definition)
        self.debug_session_id = debug_session_id
        self.debugger = get_debugger() if debug_session_id else None

    async def execute_step(
        self,
        step: dict,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Execute step with debug support"""

        if not self.debugger or not self.debug_session_id:
            return await super().execute_step(step, context)

        step_id = step.get("id")
        step_name = step.get("name", step_id)

        # Check if should break before execution
        should_break = await self.debugger.should_break(
            self.debug_session_id,
            step_id,
            context.data
        )

        if should_break:
            logger.info(f"Breaking at step: {step_name}")
            await self.debugger.pause_execution(
                self.debug_session_id,
                step_id,
                step_name,
                step.get("inputs", {}),
                context.data
            )

        # Check if execution should stop
        session = self.debugger.sessions.get(self.debug_session_id)
        if session and session.state == DebuggerState.STOPPED:
            raise Exception("Execution stopped by debugger")

        # Execute step
        result = await super().execute_step(step, context)

        # Record execution in history
        if session:
            from ia_modules.pipeline.debugger import DebugFrame
            frame = DebugFrame(
                step_id=step_id,
                step_name=step_name,
                inputs=step.get("inputs", {}),
                outputs=result,
                timestamp=asyncio.get_event_loop().time()
            )
            session.execution_history.append(frame)

        return result
```

#### 2.3 Debugger Frontend Component

```typescript
// ia_modules/web/components/PipelineDebugger/Debugger.tsx

import React, { useState, useEffect } from 'react';
import { FiPlay, FiPause, FiSkipForward, FiSquare, FiAlertCircle } from 'react-icons/fi';

interface DebuggerProps {
  sessionId: string;
  pipelineId: string;
}

interface DebugFrame {
  step_id: string;
  step_name: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  local_vars: Record<string, any>;
  timestamp: number;
}

export const PipelineDebugger: React.FC<DebuggerProps> = ({ sessionId, pipelineId }) => {
  const [state, setState] = useState<'idle' | 'running' | 'paused' | 'stopped'>('idle');
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [callStack, setCallStack] = useState<DebugFrame[]>([]);
  const [breakpoints, setBreakpoints] = useState<Set<string>>(new Set());
  const [watchExpressions, setWatchExpressions] = useState<string[]>([]);
  const [selectedVariable, setSelectedVariable] = useState<string | null>(null);

  useEffect(() => {
    // WebSocket connection for debug events
    const ws = new WebSocket(`ws://localhost:8000/api/debug/${sessionId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleDebugEvent(data);
    };

    return () => ws.close();
  }, [sessionId]);

  const handleDebugEvent = (event: any) => {
    switch (event.type) {
      case 'execution_paused':
        setState('paused');
        setCurrentStep(event.data.step_id);
        setCallStack([...callStack, event.data.frame]);
        break;
      case 'execution_resumed':
        setState('running');
        break;
      case 'execution_completed':
        setState('idle');
        break;
    }
  };

  const sendCommand = async (command: string) => {
    await fetch(`/api/debug/${sessionId}/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command }),
    });
  };

  const toggleBreakpoint = (stepId: string) => {
    const newBreakpoints = new Set(breakpoints);
    if (newBreakpoints.has(stepId)) {
      newBreakpoints.delete(stepId);
    } else {
      newBreakpoints.add(stepId);
    }
    setBreakpoints(newBreakpoints);

    fetch(`/api/debug/${sessionId}/breakpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ step_id: stepId, enabled: newBreakpoints.has(stepId) }),
    });
  };

  return (
    <div className="pipeline-debugger" style={{ display: 'flex', height: '100vh' }}>
      {/* Toolbar */}
      <div className="debugger-toolbar" style={{ padding: '10px', borderBottom: '1px solid #ccc' }}>
        <button onClick={() => sendCommand('continue')} disabled={state !== 'paused'}>
          <FiPlay /> Continue
        </button>
        <button onClick={() => sendCommand('step_over')} disabled={state !== 'paused'}>
          <FiSkipForward /> Step Over
        </button>
        <button onClick={() => sendCommand('pause')} disabled={state !== 'running'}>
          <FiPause /> Pause
        </button>
        <button onClick={() => sendCommand('stop')} disabled={state === 'idle'}>
          <FiSquare /> Stop
        </button>
      </div>

      {/* Main Content */}
      <div style={{ display: 'flex', flex: 1 }}>
        {/* Call Stack */}
        <div className="call-stack" style={{ width: '300px', borderRight: '1px solid #ccc', padding: '15px' }}>
          <h3>Call Stack</h3>
          {callStack.map((frame, index) => (
            <div
              key={index}
              className="stack-frame"
              style={{
                padding: '10px',
                marginBottom: '5px',
                background: index === callStack.length - 1 ? '#e3f2fd' : '#f5f5f5',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontWeight: 'bold' }}>{frame.step_name}</div>
              <div style={{ fontSize: '12px', color: '#666' }}>{frame.step_id}</div>
            </div>
          ))}
        </div>

        {/* Variables Inspector */}
        <div className="variables-inspector" style={{ flex: 1, padding: '15px' }}>
          <h3>Variables</h3>
          {callStack.length > 0 && (
            <div>
              <h4>Inputs</h4>
              <pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px', overflow: 'auto' }}>
                {JSON.stringify(callStack[callStack.length - 1].inputs, null, 2)}
              </pre>

              <h4>Outputs</h4>
              <pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px', overflow: 'auto' }}>
                {JSON.stringify(callStack[callStack.length - 1].outputs, null, 2)}
              </pre>

              <h4>Local Variables</h4>
              <pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px', overflow: 'auto' }}>
                {JSON.stringify(callStack[callStack.length - 1].local_vars, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Watch Panel */}
        <div className="watch-panel" style={{ width: '300px', borderLeft: '1px solid #ccc', padding: '15px' }}>
          <h3>Watch</h3>
          {watchExpressions.map((expr, index) => (
            <div key={index} style={{ marginBottom: '10px', padding: '8px', background: '#f5f5f5', borderRadius: '4px' }}>
              <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>{expr}</div>
            </div>
          ))}
          <button
            onClick={() => {
              const expr = prompt('Enter expression to watch:');
              if (expr) {
                setWatchExpressions([...watchExpressions, expr]);
              }
            }}
            style={{ marginTop: '10px', padding: '8px 12px' }}
          >
            + Add Watch
          </button>
        </div>
      </div>
    </div>
  );
};
```

#### 2.4 Debug API Routes

```python
# ia_modules/api/routes/debug.py

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import asyncio
import logging

from ia_modules.auth.middleware import get_current_user
from ia_modules.auth.models import User
from ia_modules.pipeline.debugger import get_debugger, DebuggerCommand

router = APIRouter(prefix="/api/debug", tags=["debug"])
logger = logging.getLogger(__name__)

class DebugConnectionManager:
    """Manage WebSocket connections for debugging"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_event(self, session_id: str, event: Dict[str, Any]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(event)

manager = DebugConnectionManager()

@router.websocket("/{session_id}")
async def debug_websocket(
    websocket: WebSocket,
    session_id: str
):
    """WebSocket endpoint for debug events"""
    await manager.connect(session_id, websocket)

    debugger = get_debugger()

    # Register event callback
    async def send_debug_event(event: Dict[str, Any]):
        await manager.send_event(session_id, event)

    debugger.register_event_callback(send_debug_event)

    try:
        while True:
            data = await websocket.receive_json()
            # Handle client messages if needed

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"Debug session disconnected: {session_id}")

@router.post("/{session_id}/command")
async def send_debug_command(
    session_id: str,
    command: str,
    current_user: User = Depends(get_current_user)
):
    """Send command to debugger"""
    debugger = get_debugger()

    try:
        cmd = DebuggerCommand(command)
        debugger.send_command(session_id, cmd)

        return {"message": f"Command {command} sent"}
    except ValueError:
        return {"error": f"Invalid command: {command}"}, 400

@router.post("/{session_id}/breakpoint")
async def set_breakpoint(
    session_id: str,
    step_id: str,
    enabled: bool = True,
    condition: str | None = None,
    current_user: User = Depends(get_current_user)
):
    """Set or remove breakpoint"""
    debugger = get_debugger()

    if enabled:
        debugger.set_breakpoint(session_id, step_id, condition)
    else:
        debugger.remove_breakpoint(session_id, step_id)

    return {"message": "Breakpoint updated"}

@router.get("/{session_id}/callstack")
async def get_call_stack(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get current call stack"""
    debugger = get_debugger()
    call_stack = debugger.get_call_stack(session_id)

    return {"call_stack": [frame.__dict__ for frame in call_stack]}

@router.get("/{session_id}/variable/{var_name}")
async def inspect_variable(
    session_id: str,
    var_name: str,
    current_user: User = Depends(get_current_user)
):
    """Inspect variable value"""
    debugger = get_debugger()
    value = debugger.inspect_variable(session_id, var_name)

    return {"variable": var_name, "value": value}
```

---

## 3. Mock Data Generator

### Overview
Generate realistic synthetic data for testing pipelines without real data dependencies.

### Requirements

#### 3.1 Mock Data Generator Core

```python
# ia_modules/testing/mock_data_generator.py

from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from faker import Faker
from pydantic import BaseModel
import random
import json

@dataclass
class DataGenerationConfig:
    """Configuration for data generation"""
    count: int = 10
    seed: Optional[int] = None
    locale: str = "en_US"

class MockDataGenerator:
    """Generate mock data for testing"""

    def __init__(self, config: Optional[DataGenerationConfig] = None):
        self.config = config or DataGenerationConfig()
        self.faker = Faker(self.config.locale)

        if self.config.seed:
            Faker.seed(self.config.seed)
            random.seed(self.config.seed)

    def generate_from_schema(
        self,
        schema: Dict[str, Any],
        count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate data from JSON schema"""
        count = count or self.config.count
        return [self._generate_record(schema) for _ in range(count)]

    def generate_from_pydantic(
        self,
        model: Type[BaseModel],
        count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate data from Pydantic model"""
        count = count or self.config.count
        schema = model.model_json_schema()
        return self.generate_from_schema(schema, count)

    def _generate_record(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single record from schema"""
        record = {}

        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            record[field_name] = self._generate_field(field_name, field_schema)

        return record

    def _generate_field(self, field_name: str, field_schema: Dict[str, Any]) -> Any:
        """Generate single field value"""
        field_type = field_schema.get("type", "string")
        format_type = field_schema.get("format")

        # Check for enum
        if "enum" in field_schema:
            return random.choice(field_schema["enum"])

        # Generate by type
        if field_type == "string":
            return self._generate_string(field_name, format_type)
        elif field_type == "integer":
            return self._generate_integer(field_schema)
        elif field_type == "number":
            return self._generate_number(field_schema)
        elif field_type == "boolean":
            return self.faker.boolean()
        elif field_type == "array":
            return self._generate_array(field_schema)
        elif field_type == "object":
            return self._generate_record(field_schema)
        else:
            return None

    def _generate_string(self, field_name: str, format_type: Optional[str]) -> str:
        """Generate string value based on field name and format"""

        # Format-based generation
        if format_type == "email":
            return self.faker.email()
        elif format_type == "uri":
            return self.faker.url()
        elif format_type == "date":
            return self.faker.date()
        elif format_type == "date-time":
            return self.faker.iso8601()
        elif format_type == "uuid":
            return self.faker.uuid4()

        # Field name-based generation
        field_lower = field_name.lower()

        if "email" in field_lower:
            return self.faker.email()
        elif "phone" in field_lower:
            return self.faker.phone_number()
        elif "address" in field_lower:
            return self.faker.address()
        elif "city" in field_lower:
            return self.faker.city()
        elif "country" in field_lower:
            return self.faker.country()
        elif "company" in field_lower:
            return self.faker.company()
        elif "name" in field_lower:
            if "first" in field_lower:
                return self.faker.first_name()
            elif "last" in field_lower:
                return self.faker.last_name()
            else:
                return self.faker.name()
        elif "title" in field_lower or "job" in field_lower:
            return self.faker.job()
        elif "description" in field_lower:
            return self.faker.text(max_nb_chars=200)
        elif "url" in field_lower or "website" in field_lower:
            return self.faker.url()
        elif "username" in field_lower:
            return self.faker.user_name()
        elif "password" in field_lower:
            return self.faker.password()
        elif "ip" in field_lower:
            return self.faker.ipv4()
        elif "color" in field_lower:
            return self.faker.color()
        else:
            return self.faker.word()

    def _generate_integer(self, field_schema: Dict[str, Any]) -> int:
        """Generate integer value"""
        minimum = field_schema.get("minimum", 0)
        maximum = field_schema.get("maximum", 1000)
        return random.randint(minimum, maximum)

    def _generate_number(self, field_schema: Dict[str, Any]) -> float:
        """Generate float value"""
        minimum = field_schema.get("minimum", 0.0)
        maximum = field_schema.get("maximum", 1000.0)
        return random.uniform(minimum, maximum)

    def _generate_array(self, field_schema: Dict[str, Any]) -> List[Any]:
        """Generate array value"""
        min_items = field_schema.get("minItems", 0)
        max_items = field_schema.get("maxItems", 5)
        count = random.randint(min_items, max_items)

        item_schema = field_schema.get("items", {"type": "string"})

        return [self._generate_field("item", item_schema) for _ in range(count)]

    def generate_edge_cases(self, field_type: str) -> List[Any]:
        """Generate edge case values for testing"""
        edge_cases = {
            "string": ["", " ", "a" * 1000, "🚀", None],
            "integer": [0, -1, 1, 2**31 - 1, -(2**31)],
            "number": [0.0, -0.0, float('inf'), float('-inf')],
            "boolean": [True, False],
            "array": [[], ["single"], ["a"] * 100],
        }

        return edge_cases.get(field_type, [])

    def generate_volume_data(
        self,
        schema: Dict[str, Any],
        count: int = 10000,
        batch_size: int = 1000
    ):
        """Generate large volume of data in batches"""
        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            yield self.generate_from_schema(schema, batch_count)

# Predefined templates
class DataTemplates:
    """Common data templates"""

    @staticmethod
    def user_profile() -> Dict[str, Any]:
        """User profile schema"""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "email": {"type": "string", "format": "email"},
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "phone": {"type": "string"},
                "address": {"type": "string"},
                "city": {"type": "string"},
                "country": {"type": "string"},
                "age": {"type": "integer", "minimum": 18, "maximum": 100},
                "created_at": {"type": "string", "format": "date-time"},
            }
        }

    @staticmethod
    def product() -> Dict[str, Any]:
        """Product schema"""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "price": {"type": "number", "minimum": 0.0, "maximum": 10000.0},
                "category": {"type": "string", "enum": ["Electronics", "Clothing", "Food", "Books"]},
                "in_stock": {"type": "boolean"},
                "quantity": {"type": "integer", "minimum": 0, "maximum": 1000},
            }
        }

    @staticmethod
    def transaction() -> Dict[str, Any]:
        """Transaction schema"""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "user_id": {"type": "string", "format": "uuid"},
                "amount": {"type": "number", "minimum": 0.01, "maximum": 100000.0},
                "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                "status": {"type": "string", "enum": ["pending", "completed", "failed"]},
                "timestamp": {"type": "string", "format": "date-time"},
            }
        }
```

#### 3.2 Mock Data CLI

```python
# ia_modules/cli/mock_data.py

import click
import json
from ia_modules.testing.mock_data_generator import MockDataGenerator, DataGenerationConfig, DataTemplates

@click.group()
def mock_data():
    """Mock data generation commands"""
    pass

@mock_data.command()
@click.option('--schema', '-s', required=True, help='Path to JSON schema file')
@click.option('--count', '-c', default=10, help='Number of records to generate')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def generate(schema: str, count: int, output: str, format: str, seed: int):
    """Generate mock data from schema"""

    # Load schema
    with open(schema, 'r') as f:
        schema_dict = json.load(f)

    # Generate data
    config = DataGenerationConfig(count=count, seed=seed)
    generator = MockDataGenerator(config)

    data = generator.generate_from_schema(schema_dict, count)

    # Output
    if format == 'json':
        output_data = json.dumps(data, indent=2)
    elif format == 'csv':
        import csv
        import io

        output_io = io.StringIO()
        if data:
            writer = csv.DictWriter(output_io, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        output_data = output_io.getvalue()

    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Generated {count} records to {output}")
    else:
        click.echo(output_data)

@mock_data.command()
@click.argument('template', type=click.Choice(['user', 'product', 'transaction']))
@click.option('--count', '-c', default=10)
@click.option('--output', '-o')
def from_template(template: str, count: int, output: str):
    """Generate mock data from predefined template"""

    # Get template schema
    template_map = {
        'user': DataTemplates.user_profile(),
        'product': DataTemplates.product(),
        'transaction': DataTemplates.transaction(),
    }

    schema = template_map[template]

    # Generate
    generator = MockDataGenerator(DataGenerationConfig(count=count))
    data = generator.generate_from_schema(schema, count)

    output_data = json.dumps(data, indent=2)

    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Generated {count} {template} records to {output}")
    else:
        click.echo(output_data)

@mock_data.command()
@click.option('--type', '-t', required=True, help='Field type')
def edge_cases(type: str):
    """Generate edge case values for type"""
    generator = MockDataGenerator()
    cases = generator.generate_edge_cases(type)

    click.echo(json.dumps(cases, indent=2))
```

---

## 4. Advanced CLI Tools

### Overview
Rich terminal user interface with interactive features, auto-completion, and beautiful output.

### Requirements

#### 4.1 Rich CLI Framework

```python
# ia_modules/cli/rich_cli.py

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
import json

console = Console()

@click.group()
def cli():
    """IA Modules CLI - AI Pipeline Orchestration"""
    pass

@cli.command()
@click.option('--pipeline-id', '-p', help='Pipeline ID')
@click.option('--status', '-s', type=click.Choice(['running', 'completed', 'failed', 'all']), default='all')
def list_executions(pipeline_id: str, status: str):
    """List pipeline executions"""

    # Fetch executions (mock data for example)
    executions = [
        {"id": "exec-001", "pipeline": "Data Processing", "status": "completed", "duration": "2m 30s"},
        {"id": "exec-002", "pipeline": "ML Training", "status": "running", "duration": "10m 15s"},
        {"id": "exec-003", "pipeline": "API Integration", "status": "failed", "duration": "30s"},
    ]

    # Filter by status
    if status != 'all':
        executions = [e for e in executions if e['status'] == status]

    # Create table
    table = Table(title="Pipeline Executions", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Pipeline", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right", style="green")

    for exec in executions:
        # Color code status
        status_color = {
            "completed": "[green]✓ completed[/green]",
            "running": "[yellow]⟳ running[/yellow]",
            "failed": "[red]✗ failed[/red]",
        }.get(exec['status'], exec['status'])

        table.add_row(
            exec['id'],
            exec['pipeline'],
            status_color,
            exec['duration']
        )

    console.print(table)

@cli.command()
@click.argument('execution-id')
def logs(execution_id: str):
    """View execution logs with syntax highlighting"""

    # Mock log data
    log_lines = [
        "[2025-10-25 10:30:00] INFO: Starting pipeline execution",
        "[2025-10-25 10:30:01] DEBUG: Loading step definitions",
        "[2025-10-25 10:30:02] INFO: Executing step 1: Data Ingestion",
        "[2025-10-25 10:30:05] WARNING: Large dataset detected, may take longer",
        "[2025-10-25 10:30:10] INFO: Step 1 completed successfully",
        "[2025-10-25 10:30:11] ERROR: Step 2 failed: Connection timeout",
    ]

    console.print(Panel(f"Logs for execution: {execution_id}", style="bold blue"))

    for line in log_lines:
        # Color code log levels
        if "ERROR" in line:
            console.print(line, style="red")
        elif "WARNING" in line:
            console.print(line, style="yellow")
        elif "INFO" in line:
            console.print(line, style="green")
        elif "DEBUG" in line:
            console.print(line, style="dim")
        else:
            console.print(line)

@cli.command()
@click.argument('pipeline-file')
def validate(pipeline_file: str):
    """Validate pipeline definition"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Validating pipeline...", total=100)

        # Mock validation steps
        import time

        progress.update(task, advance=25, description="Parsing JSON...")
        time.sleep(0.5)

        progress.update(task, advance=25, description="Checking schema...")
        time.sleep(0.5)

        progress.update(task, advance=25, description="Validating steps...")
        time.sleep(0.5)

        progress.update(task, advance=25, description="Validation complete!")

    console.print("\n[green]✓[/green] Pipeline validation successful!", style="bold green")

@cli.command()
@click.option('--name', prompt='Pipeline name')
@click.option('--description', prompt='Description')
def create(name: str, description: str):
    """Interactive pipeline creation wizard"""

    console.print(Panel("Pipeline Creation Wizard", style="bold blue"))

    # Step type selection
    step_type = Prompt.ask(
        "Select step type",
        choices=["llm", "transform", "condition", "integration", "loop"],
        default="llm"
    )

    # Module path
    module = Prompt.ask("Enter module path", default=f"ia_modules.pipeline.steps.{step_type}_step")

    # Inputs
    console.print("\n[bold]Define inputs:[/bold]")
    inputs = {}
    while True:
        add_input = Confirm.ask("Add input field?")
        if not add_input:
            break

        input_name = Prompt.ask("Input name")
        input_type = Prompt.ask("Input type", choices=["string", "number", "boolean", "object"], default="string")
        inputs[input_name] = {"type": input_type}

    # Build pipeline definition
    pipeline = {
        "name": name,
        "description": description,
        "steps": [
            {
                "id": "step-1",
                "name": f"{step_type.title()} Step",
                "type": step_type,
                "module": module,
                "inputs": inputs
            }
        ]
    }

    # Display preview
    console.print("\n[bold]Pipeline Preview:[/bold]")
    syntax = Syntax(json.dumps(pipeline, indent=2), "json", theme="monokai")
    console.print(syntax)

    # Save
    if Confirm.ask("\nSave pipeline?"):
        filename = f"{name.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(pipeline, f, indent=2)
        console.print(f"\n[green]✓[/green] Pipeline saved to {filename}", style="bold green")

@cli.command()
@click.argument('pipeline-file')
def visualize(pipeline_file: str):
    """Visualize pipeline as tree"""

    with open(pipeline_file, 'r') as f:
        pipeline = json.load(f)

    tree = Tree(f"[bold blue]{pipeline.get('name', 'Pipeline')}[/bold blue]")

    for step in pipeline.get('steps', []):
        step_node = tree.add(f"[green]{step.get('name', 'Unnamed Step')}[/green]")
        step_node.add(f"[dim]Type: {step.get('type', 'unknown')}[/dim]")
        step_node.add(f"[dim]ID: {step.get('id', 'unknown')}[/dim]")

        if step.get('inputs'):
            inputs_node = step_node.add("[yellow]Inputs[/yellow]")
            for key in step['inputs'].keys():
                inputs_node.add(f"• {key}")

    console.print(tree)

if __name__ == '__main__':
    cli()
```

#### 4.2 Auto-Completion Support

```python
# ia_modules/cli/completion.py

import click
from click.shell_completion import CompletionItem

def get_pipeline_names(ctx, param, incomplete):
    """Get pipeline names for auto-completion"""
    # In real implementation, fetch from database
    pipelines = [
        "data-processing-pipeline",
        "ml-training-pipeline",
        "api-integration-pipeline",
        "data-validation-pipeline",
    ]

    return [
        CompletionItem(name, help=f"Pipeline: {name}")
        for name in pipelines
        if incomplete in name
    ]

def get_execution_ids(ctx, param, incomplete):
    """Get execution IDs for auto-completion"""
    # In real implementation, fetch from database
    executions = [
        "exec-001",
        "exec-002",
        "exec-003",
    ]

    return [
        CompletionItem(eid)
        for eid in executions
        if incomplete in eid
    ]

@click.command()
@click.argument('pipeline', shell_complete=get_pipeline_names)
@click.argument('execution-id', shell_complete=get_execution_ids)
def run(pipeline: str, execution_id: str):
    """Run pipeline with auto-completion"""
    click.echo(f"Running {pipeline} with execution ID {execution_id}")

# Generate completion script
def generate_completion_script(shell: str = "bash"):
    """Generate shell completion script"""

    scripts = {
        "bash": """
# ia-modules bash completion
_ia_modules_completion() {
    local IFS=$'\\n'
    COMPREPLY=( $(env COMP_WORDS="${COMP_WORDS[*]}" \\
                     COMP_CWORD=$COMP_CWORD \\
                     _IA_MODULES_COMPLETE=complete $1) )
}

complete -F _ia_modules_completion -o default ia-modules
""",
        "zsh": """
# ia-modules zsh completion
#compdef ia-modules

_ia_modules() {
    eval $(env COMMANDLINE="${words[1,$CURRENT]}" _IA_MODULES_COMPLETE=complete-zsh ia-modules)
}

if [[ "$(basename ${(%):-%x})" != "_ia-modules" ]]; then
    compdef _ia_modules ia-modules
fi
""",
    }

    return scripts.get(shell, "")
```

---

## 5. IDE Integration

### Overview
VS Code extension with language server for pipeline development support.

### Requirements

#### 5.1 VS Code Extension

```typescript
// ia_modules/vscode-extension/src/extension.ts

import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';
import * as path from 'path';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('IA Modules extension activated');

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('ia-modules.validatePipeline', validatePipeline),
        vscode.commands.registerCommand('ia-modules.runPipeline', runPipeline),
        vscode.commands.registerCommand('ia-modules.debugPipeline', debugPipeline),
        vscode.commands.registerCommand('ia-modules.visualizePipeline', visualizePipeline)
    );

    // Start language server
    startLanguageServer(context);

    // Register code lens provider
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { scheme: 'file', pattern: '**/*pipeline*.json' },
            new PipelineCodeLensProvider()
        )
    );

    // Register completion provider
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            { scheme: 'file', pattern: '**/*pipeline*.json' },
            new PipelineCompletionProvider(),
            '.'
        )
    );

    // Register hover provider
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(
            { scheme: 'file', pattern: '**/*pipeline*.json' },
            new PipelineHoverProvider()
        )
    );
}

function startLanguageServer(context: vscode.ExtensionContext) {
    const serverModule = context.asAbsolutePath(
        path.join('server', 'out', 'server.js')
    );

    const serverOptions: ServerOptions = {
        run: { module: serverModule, transport: TransportKind.ipc },
        debug: {
            module: serverModule,
            transport: TransportKind.ipc,
            options: { execArgv: ['--nolazy', '--inspect=6009'] }
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', pattern: '**/*pipeline*.json' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*pipeline*.json')
        }
    };

    client = new LanguageClient(
        'ia-modules-language-server',
        'IA Modules Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
}

async function validatePipeline() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return;
    }

    const document = editor.document;
    const text = document.getText();

    try {
        const pipeline = JSON.parse(text);

        // Call validation API
        const response = await fetch('http://localhost:8000/api/designer/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                nodes: pipeline.steps || [],
                edges: []
            })
        });

        const result = await response.json();

        if (result.valid) {
            vscode.window.showInformationMessage('✓ Pipeline is valid');
        } else {
            vscode.window.showErrorMessage(
                `Pipeline validation failed: ${result.errors.join(', ')}`
            );
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Validation error: ${error.message}`);
    }
}

async function runPipeline() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return;
    }

    const pipelineFile = editor.document.fileName;

    // Show terminal and run pipeline
    const terminal = vscode.window.createTerminal('IA Modules');
    terminal.show();
    terminal.sendText(`ia-modules run ${pipelineFile}`);
}

async function debugPipeline() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return;
    }

    // Start debug session
    vscode.debug.startDebugging(undefined, {
        type: 'ia-modules',
        request: 'launch',
        name: 'Debug Pipeline',
        pipelineFile: editor.document.fileName
    });
}

async function visualizePipeline() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return;
    }

    const document = editor.document;
    const text = document.getText();
    const pipeline = JSON.parse(text);

    // Open webview panel for visualization
    const panel = vscode.window.createWebviewPanel(
        'pipelineVisualization',
        'Pipeline Visualization',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true
        }
    );

    panel.webview.html = getVisualizationHtml(pipeline);
}

function getVisualizationHtml(pipeline: any): string {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Pipeline Visualization</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { margin: 0; padding: 20px; }
                .node { fill: #69b3a2; stroke: #fff; stroke-width: 2px; }
                .link { stroke: #999; stroke-width: 2px; }
                text { font-family: sans-serif; font-size: 12px; }
            </style>
        </head>
        <body>
            <div id="visualization"></div>
            <script>
                const pipeline = ${JSON.stringify(pipeline)};
                // D3.js visualization code here
            </script>
        </body>
        </html>
    `;
}

class PipelineCodeLensProvider implements vscode.CodeLensProvider {
    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
        const codeLenses: vscode.CodeLens[] = [];

        // Add "Run Pipeline" code lens at top
        const topLine = document.lineAt(0);
        codeLenses.push(
            new vscode.CodeLens(topLine.range, {
                title: '▶ Run Pipeline',
                command: 'ia-modules.runPipeline'
            })
        );

        codeLenses.push(
            new vscode.CodeLens(topLine.range, {
                title: '🐛 Debug Pipeline',
                command: 'ia-modules.debugPipeline'
            })
        );

        codeLenses.push(
            new vscode.CodeLens(topLine.range, {
                title: '📊 Visualize',
                command: 'ia-modules.visualizePipeline'
            })
        );

        return codeLenses;
    }
}

class PipelineCompletionProvider implements vscode.CompletionItemProvider {
    provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position
    ): vscode.CompletionItem[] {
        const completions: vscode.CompletionItem[] = [];

        // Add step type completions
        const stepTypes = ['llm', 'transform', 'condition', 'integration', 'loop'];
        stepTypes.forEach(type => {
            const completion = new vscode.CompletionItem(type, vscode.CompletionItemKind.Value);
            completion.detail = `${type} step`;
            completion.insertText = `"${type}"`;
            completions.push(completion);
        });

        return completions;
    }
}

class PipelineHoverProvider implements vscode.HoverProvider {
    provideHover(
        document: vscode.TextDocument,
        position: vscode.Position
    ): vscode.Hover | null {
        const range = document.getWordRangeAtPosition(position);
        const word = document.getText(range);

        const stepDocs = {
            'llm': 'Execute LLM prompt with configurable model',
            'transform': 'Transform data using Python function',
            'condition': 'Branch execution based on condition',
            'integration': 'Integrate with external services',
            'loop': 'Iterate over collection'
        };

        if (stepDocs[word]) {
            return new vscode.Hover(stepDocs[word]);
        }

        return null;
    }
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
```

#### 5.2 Pipeline JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IA Modules Pipeline",
  "type": "object",
  "required": ["name", "steps"],
  "properties": {
    "name": {
      "type": "string",
      "description": "Pipeline name"
    },
    "description": {
      "type": "string",
      "description": "Pipeline description"
    },
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "description": "Pipeline version (semver)"
    },
    "steps": {
      "type": "array",
      "description": "Pipeline steps",
      "items": {
        "$ref": "#/definitions/step"
      }
    }
  },
  "definitions": {
    "step": {
      "type": "object",
      "required": ["id", "name", "type", "module"],
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique step identifier"
        },
        "name": {
          "type": "string",
          "description": "Step name"
        },
        "type": {
          "type": "string",
          "enum": ["llm", "transform", "condition", "integration", "loop", "parallel"],
          "description": "Step type"
        },
        "module": {
          "type": "string",
          "description": "Python module path"
        },
        "inputs": {
          "type": "object",
          "description": "Step input configuration"
        },
        "outputs": {
          "type": "object",
          "description": "Step output mapping"
        },
        "next_steps": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Next step IDs"
        },
        "error_handler": {
          "type": "string",
          "description": "Error handler step ID"
        }
      }
    }
  }
}
```

#### 5.3 Debug Adapter Protocol

```typescript
// ia_modules/vscode-extension/src/debugAdapter.ts

import {
    DebugSession,
    InitializedEvent,
    TerminatedEvent,
    StoppedEvent,
    BreakpointEvent,
    OutputEvent,
    Thread,
    StackFrame,
    Scope,
    Source,
    Handles
} from 'vscode-debugadapter';
import { DebugProtocol } from 'vscode-debugprotocol';
import * as WebSocket from 'ws';

interface LaunchRequestArguments extends DebugProtocol.LaunchRequestArguments {
    pipelineFile: string;
}

export class PipelineDebugSession extends DebugSession {
    private ws: WebSocket;
    private variableHandles = new Handles<string>();
    private breakpoints = new Map<string, DebugProtocol.Breakpoint[]>();

    public constructor() {
        super();
    }

    protected initializeRequest(
        response: DebugProtocol.InitializeResponse,
        args: DebugProtocol.InitializeRequestArguments
    ): void {
        response.body = response.body || {};
        response.body.supportsConfigurationDoneRequest = true;
        response.body.supportsBreakpointLocationsRequest = true;
        response.body.supportsEvaluateForHovers = true;
        response.body.supportsStepBack = false;

        this.sendResponse(response);
        this.sendEvent(new InitializedEvent());
    }

    protected async launchRequest(
        response: DebugProtocol.LaunchResponse,
        args: LaunchRequestArguments
    ): Promise<void> {
        // Connect to debug backend via WebSocket
        this.ws = new WebSocket('ws://localhost:8000/api/debug/session-123');

        this.ws.on('message', (data) => {
            const event = JSON.parse(data.toString());
            this.handleDebugEvent(event);
        });

        this.ws.on('close', () => {
            this.sendEvent(new TerminatedEvent());
        });

        // Start pipeline execution in debug mode
        await this.startPipelineDebug(args.pipelineFile);

        this.sendResponse(response);
    }

    protected setBreakPointsRequest(
        response: DebugProtocol.SetBreakpointsResponse,
        args: DebugProtocol.SetBreakpointsArguments
    ): void {
        const path = args.source.path!;
        const breakpoints = args.breakpoints || [];

        // Send breakpoints to backend
        breakpoints.forEach((bp) => {
            this.ws.send(JSON.stringify({
                command: 'set_breakpoint',
                step_id: `step-${bp.line}`,
                condition: bp.condition
            }));
        });

        const actualBreakpoints = breakpoints.map(bp => ({
            verified: true,
            line: bp.line
        }));

        response.body = {
            breakpoints: actualBreakpoints
        };

        this.sendResponse(response);
    }

    protected threadsRequest(response: DebugProtocol.ThreadsResponse): void {
        response.body = {
            threads: [new Thread(1, 'Pipeline Execution')]
        };
        this.sendResponse(response);
    }

    protected stackTraceRequest(
        response: DebugProtocol.StackTraceResponse,
        args: DebugProtocol.StackTraceArguments
    ): void {
        // Request call stack from backend
        this.ws.send(JSON.stringify({
            command: 'get_callstack'
        }));

        // For now, return mock stack
        response.body = {
            stackFrames: [
                new StackFrame(
                    0,
                    'Step 1: Data Ingestion',
                    new Source('pipeline.json', 'pipeline.json'),
                    10
                )
            ],
            totalFrames: 1
        };

        this.sendResponse(response);
    }

    protected scopesRequest(
        response: DebugProtocol.ScopesResponse,
        args: DebugProtocol.ScopesArguments
    ): void {
        response.body = {
            scopes: [
                new Scope('Inputs', this.variableHandles.create('inputs'), false),
                new Scope('Outputs', this.variableHandles.create('outputs'), false),
                new Scope('Context', this.variableHandles.create('context'), false)
            ]
        };

        this.sendResponse(response);
    }

    protected continueRequest(
        response: DebugProtocol.ContinueResponse,
        args: DebugProtocol.ContinueArguments
    ): void {
        this.ws.send(JSON.stringify({ command: 'continue' }));
        this.sendResponse(response);
    }

    protected nextRequest(
        response: DebugProtocol.NextResponse,
        args: DebugProtocol.NextArguments
    ): void {
        this.ws.send(JSON.stringify({ command: 'step_over' }));
        this.sendResponse(response);
    }

    protected evaluateRequest(
        response: DebugProtocol.EvaluateResponse,
        args: DebugProtocol.EvaluateArguments
    ): void {
        // Evaluate expression in pipeline context
        this.ws.send(JSON.stringify({
            command: 'evaluate',
            expression: args.expression
        }));

        // Mock response
        response.body = {
            result: 'evaluation result',
            variablesReference: 0
        };

        this.sendResponse(response);
    }

    private handleDebugEvent(event: any): void {
        switch (event.type) {
            case 'execution_paused':
                this.sendEvent(new StoppedEvent('breakpoint', 1));
                break;
            case 'output':
                this.sendEvent(new OutputEvent(event.data.message));
                break;
        }
    }

    private async startPipelineDebug(pipelineFile: string): Promise<void> {
        // Start pipeline in debug mode via API
        await fetch('http://localhost:8000/api/pipelines/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pipeline_file: pipelineFile,
                debug_mode: true,
                debug_session_id: 'session-123'
            })
        });
    }
}
```

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- [ ] Visual Pipeline Designer core components
- [ ] Basic React Flow integration
- [ ] Step library implementation
- [ ] Properties panel

### Phase 2: Debugging (Weeks 3-4)
- [ ] Debugger core engine
- [ ] Breakpoint management
- [ ] WebSocket debug protocol
- [ ] Debug frontend UI

### Phase 3: Testing Tools (Weeks 5-6)
- [ ] Mock data generator core
- [ ] Schema-based generation
- [ ] CLI commands for data generation
- [ ] Template library

### Phase 4: CLI Enhancement (Weeks 7-8)
- [ ] Rich CLI framework
- [ ] Interactive commands
- [ ] Auto-completion
- [ ] Progress indicators

### Phase 5: IDE Integration (Weeks 9-10)
- [ ] VS Code extension
- [ ] Language server
- [ ] JSON schema validation
- [ ] Debug adapter protocol

### Phase 6: Polish & Documentation (Weeks 11-12)
- [ ] User documentation
- [ ] Video tutorials
- [ ] Example pipelines
- [ ] Performance optimization

---

## Dependencies & Prerequisites

### Python Dependencies
```
# ia_modules/requirements-dev.txt
rich>=13.0.0
click>=8.1.0
faker>=20.0.0
click-completion>=0.5.2
websockets>=11.0.0
```

### TypeScript/Node Dependencies
```json
{
  "dependencies": {
    "reactflow": "^11.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "dagre": "^0.8.5",
    "react-icons": "^4.11.0",
    "vscode-languageclient": "^8.1.0",
    "vscode-debugadapter": "^1.51.0"
  }
}
```

### System Requirements
- Python 3.11+
- Node.js 18+
- VS Code 1.80+
- Modern web browser with WebSocket support

---

## Success Criteria

### Visual Designer
- [ ] Can create pipelines without writing JSON
- [ ] Real-time validation with error highlighting
- [ ] Export to valid JSON format
- [ ] Import existing pipelines
- [ ] Auto-layout algorithm works correctly

### Debugger
- [ ] Can set/remove breakpoints
- [ ] Step-through execution works
- [ ] Variable inspection shows correct values
- [ ] WebSocket communication stable
- [ ] Time-travel debugging functional

### Mock Data Generator
- [ ] Generates realistic data from schemas
- [ ] Supports all common data types
- [ ] Edge case generation works
- [ ] Performance acceptable for large volumes
- [ ] CLI integration seamless

### CLI Tools
- [ ] Beautiful, colorful output
- [ ] Auto-completion works in bash/zsh
- [ ] Interactive prompts user-friendly
- [ ] Progress indicators smooth
- [ ] Help text comprehensive

### IDE Integration
- [ ] Extension installs without errors
- [ ] IntelliSense provides useful suggestions
- [ ] Syntax validation catches errors
- [ ] Debug adapter works correctly
- [ ] Code lens actions functional

---

**End of Developer Experience Implementation Plan**

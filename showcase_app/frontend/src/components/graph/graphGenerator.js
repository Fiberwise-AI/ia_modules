import { calculateLevels, buildAdjacencyList, groupByLevel, calculateNodePosition } from './utils/graphLayout'
import { detectNodeType } from './utils/nodeTypeDetector'
import { formatEdge } from './utils/edgeFormatter'

/**
 * Generate graph nodes and edges from pipeline configuration
 * Supports both 'paths' and 'transitions' flow formats
 */
export function generateGraph(pipeline) {
  const config = pipeline.config
  const steps = config.steps || []
  const flow = config.flow || {}

  // Normalize flow format (handle both 'paths' and 'transitions')
  const paths = normalizePaths(flow)

  if (paths.length === 0) {
    return generateSequentialGraph(steps)
  }

  return generateFlowGraph(steps, { ...flow, paths })
}

/**
 * Normalize different flow path formats
 */
function normalizePaths(flow) {
  if (flow.paths) return flow.paths

  if (flow.transitions) {
    // Convert transitions format to paths format
    return flow.transitions.map(t => ({
      from_step: t.from,
      to_step: t.to,
      condition: t.condition
    }))
  }

  return []
}

/**
 * Generate sequential graph (fallback when no flow defined)
 */
function generateSequentialGraph(steps) {
  const nodes = steps.map((step, idx) => ({
    id: step.id || `step${idx + 1}`,
    type: 'step',
    position: { x: 250, y: idx * 120 },
    data: { label: step.name, status: 'pending' },
  }))

  const edges = []
  for (let i = 0; i < steps.length - 1; i++) {
    edges.push({
      id: `e${i}-${i + 1}`,
      source: nodes[i].id,
      target: nodes[i + 1].id,
      animated: true,
    })
  }

  return { nodes, edges }
}

/**
 * Generate graph from flow definition
 */
function generateFlowGraph(steps, flow) {
  const stepMap = createStepMap(steps)
  const { outgoing, incoming } = buildAdjacencyList(flow.paths)
  const levels = calculateLevels(flow, steps)
  const stepsByLevel = groupByLevel(levels)

  const nodes = generateNodes(stepsByLevel, stepMap, outgoing, incoming)
  const edges = generateEdges(flow.paths)

  return { nodes, edges }
}

/**
 * Create step ID to step object lookup map
 */
function createStepMap(steps) {
  const map = {}
  steps.forEach(step => {
    map[step.id] = step
  })
  return map
}

/**
 * Generate nodes from grouped steps
 */
function generateNodes(stepsByLevel, stepMap, outgoing, incoming) {
  const nodes = []

  Object.keys(stepsByLevel).forEach(level => {
    const stepsAtLevel = stepsByLevel[level]
    const levelNum = parseInt(level)

    stepsAtLevel.forEach((stepId, idx) => {
      const step = stepMap[stepId]
      if (!step) return

      const position = calculateNodePosition(
        levelNum,
        idx,
        stepsAtLevel.length
      )

      const nodeType = detectNodeType(stepId, outgoing, incoming)

      nodes.push({
        id: stepId,
        type: nodeType,
        position,
        data: {
          label: step.name,
          status: 'pending',
        },
      })
    })
  })

  return nodes
}

/**
 * Generate edges from flow paths
 */
function generateEdges(paths) {
  return paths.map(path => {
    const edgeId = `e-${path.from_step}-${path.to_step}`
    const { label, style } = formatEdge(path)

    return {
      id: edgeId,
      source: path.from_step,
      target: path.to_step,
      label,
      style,
      animated: true,
    }
  })
}

/**
 * Format edge with appropriate label and style based on condition
 */
export function formatEdge(path) {
  let label = undefined
  let style = {}

  if (path.condition && path.condition.type !== 'always') {
    if (path.condition.type === 'threshold_condition') {
      const config = path.condition.config
      const operator = config?.operator || ''

      if (operator === '>=') {
        label = 'HIGH'
        style = { stroke: '#10b981', strokeWidth: 2 }
      } else if (operator === '<') {
        label = 'LOW'
        style = { stroke: '#ef4444', strokeWidth: 2 }
      } else {
        const field = config?.field || ''
        const value = config?.value || ''
        label = `${field} ${operator} ${value}`
      }
    } else {
      label = path.condition.type
    }
  }

  return { label, style }
}

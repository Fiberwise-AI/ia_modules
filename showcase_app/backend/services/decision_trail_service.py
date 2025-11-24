"""
Decision Trail Service

Wraps ia_modules decision trail functionality for the showcase app.
Provides access to decision logging, execution paths, and evidence tracking.
"""

from typing import Dict, Any, List


class DecisionTrailService:
    """Service for managing decision trails and execution paths"""

    def __init__(self, decision_trail_builder, reliability_metrics):
        """
        Initialize decision trail service
        
        Args:
            decision_trail_builder: ia_modules DecisionTrailBuilder instance
            reliability_metrics: ReliabilityMetrics service for context
        """
        self.decision_trail_builder = decision_trail_builder
        self.reliability_metrics = reliability_metrics

    async def get_decision_trail(self, job_id: str) -> Dict[str, Any]:
        """
        Get complete decision trail for an execution
        
        Args:
            job_id: Execution job ID
            
        Returns:
            Decision trail with nodes, edges, and metadata
        """
        try:
            # Get decision trail from builder if available
            trail = None
            if self.decision_trail_builder:
                trail = self.decision_trail_builder.get_trail(job_id)
            
            if not trail:
                return {
                    "job_id": job_id,
                    "nodes": [],
                    "edges": [],
                    "metadata": {},
                    "statistics": {
                        "total_decisions": 0,
                        "decision_points": 0,
                        "paths_taken": 0
                    }
                }
            
            # Format trail data
            nodes = self._format_decision_nodes(trail.get("nodes", []))
            edges = self._format_decision_edges(trail.get("edges", []))
            
            return {
                "job_id": job_id,
                "nodes": nodes,
                "edges": edges,
                "metadata": trail.get("metadata", {}),
                "statistics": self._calculate_trail_statistics(nodes, edges),
                "created_at": trail.get("created_at"),
                "updated_at": trail.get("updated_at")
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get decision trail: {str(e)}")

    async def get_decision_node(self, job_id: str, node_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific decision node
        
        Args:
            job_id: Execution job ID
            node_id: Decision node ID
            
        Returns:
            Decision node details with evidence and rationale
        """
        try:
            node = self.decision_trail_builder.get_node(job_id, node_id)
            
            if not node:
                raise ValueError(f"Decision node not found: {node_id}")
            
            return {
                "node_id": node_id,
                "job_id": job_id,
                "decision_type": node.get("decision_type"),
                "decision": node.get("decision"),
                "rationale": node.get("rationale"),
                "confidence": node.get("confidence"),
                "evidence": node.get("evidence", []),
                "alternatives": node.get("alternatives", []),
                "timestamp": node.get("timestamp"),
                "metadata": node.get("metadata", {})
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get decision node: {str(e)}")

    async def get_execution_path(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get the execution path taken through decision points
        
        Args:
            job_id: Execution job ID
            
        Returns:
            Ordered list of decisions and their outcomes
        """
        try:
            path = self.decision_trail_builder.get_execution_path(job_id)
            
            return [
                {
                    "step": idx + 1,
                    "node_id": decision["node_id"],
                    "decision_type": decision.get("decision_type"),
                    "decision": decision.get("decision"),
                    "outcome": decision.get("outcome"),
                    "timestamp": decision.get("timestamp")
                }
                for idx, decision in enumerate(path or [])
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get execution path: {str(e)}")

    async def get_decision_evidence(self, job_id: str, node_id: str) -> List[Dict[str, Any]]:
        """
        Get evidence collected for a specific decision
        
        Args:
            job_id: Execution job ID
            node_id: Decision node ID
            
        Returns:
            List of evidence items with sources and weights
        """
        try:
            evidence = self.decision_trail_builder.get_evidence(job_id, node_id)
            
            return [
                {
                    "evidence_id": item.get("id"),
                    "type": item.get("type"),  # direct, inferred, contextual
                    "source": item.get("source"),
                    "content": item.get("content"),
                    "weight": item.get("weight", 1.0),
                    "confidence": item.get("confidence", 1.0),
                    "timestamp": item.get("timestamp")
                }
                for item in (evidence or [])
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get decision evidence: {str(e)}")

    async def get_alternative_paths(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get alternative decision paths that were not taken
        
        Args:
            job_id: Execution job ID
            
        Returns:
            List of alternative paths with their probabilities
        """
        try:
            alternatives = self.decision_trail_builder.get_alternatives(job_id)
            
            return [
                {
                    "path_id": alt.get("path_id"),
                    "decision_point": alt.get("decision_point"),
                    "alternative_decision": alt.get("decision"),
                    "probability": alt.get("probability", 0.0),
                    "reason_not_taken": alt.get("reason"),
                    "potential_outcome": alt.get("outcome")
                }
                for alt in (alternatives or [])
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to get alternative paths: {str(e)}")

    async def export_trail(self, job_id: str, format: str = "json") -> Dict[str, Any]:
        """
        Export decision trail in specified format
        
        Args:
            job_id: Execution job ID
            format: Export format (json, graphviz, mermaid)
            
        Returns:
            Exported trail data
        """
        try:
            trail = await self.get_decision_trail(job_id)
            
            if format == "json":
                return trail
            elif format == "graphviz":
                return self._export_graphviz(trail)
            elif format == "mermaid":
                return self._export_mermaid(trail)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to export trail: {str(e)}")

    def _format_decision_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format decision nodes for API response"""
        return [
            {
                "id": node["id"],
                "type": node.get("type", "decision"),
                "label": node.get("label", node["id"]),
                "decision": node.get("decision"),
                "confidence": node.get("confidence", 1.0),
                "timestamp": node.get("timestamp"),
                "metadata": node.get("metadata", {})
            }
            for node in nodes
        ]

    def _format_decision_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format decision edges for API response"""
        return [
            {
                "from": edge["from"],
                "to": edge["to"],
                "label": edge.get("label", ""),
                "condition": edge.get("condition"),
                "weight": edge.get("weight", 1.0)
            }
            for edge in edges
        ]

    def _calculate_trail_statistics(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for decision trail"""
        decision_nodes = [n for n in nodes if n.get("type") == "decision"]
        
        return {
            "total_nodes": len(nodes),
            "decision_points": len(decision_nodes),
            "total_edges": len(edges),
            "paths_taken": len([e for e in edges if e.get("weight", 0) > 0]),
            "average_confidence": sum(n.get("confidence", 0) for n in decision_nodes) / max(len(decision_nodes), 1)
        }

    def _export_graphviz(self, trail: Dict[str, Any]) -> str:
        """Export trail as Graphviz DOT format"""
        lines = ["digraph DecisionTrail {", "  rankdir=LR;"]
        
        # Add nodes
        for node in trail["nodes"]:
            label = f"{node['label']}\n{node.get('decision', '')}"
            lines.append(f'  "{node["id"]}" [label="{label}"];')
        
        # Add edges
        for edge in trail["edges"]:
            label = edge.get("label", "")
            lines.append(f'  "{edge["from"]}" -> "{edge["to"]}" [label="{label}"];')
        
        lines.append("}")
        return "\n".join(lines)

    def _export_mermaid(self, trail: Dict[str, Any]) -> str:
        """Export trail as Mermaid diagram"""
        lines = ["graph LR"]
        
        # Add nodes
        for node in trail["nodes"]:
            label = f"{node['label']}: {node.get('decision', '')}"
            lines.append(f'  {node["id"]}["{label}"]')
        
        # Add edges
        for edge in trail["edges"]:
            label = edge.get("label", "")
            if label:
                lines.append(f'  {edge["from"]} -->|{label}| {edge["to"]}')
            else:
                lines.append(f'  {edge["from"]} --> {edge["to"]}')
        
        return "\n".join(lines)

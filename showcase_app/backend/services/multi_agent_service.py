"""
Multi-Agent Orchestration Service

Demonstrates multi-agent workflows with communication tracking,
state management, and coordination patterns.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, UTC
import asyncio
import json
import os
from pathlib import Path
from ia_modules.agents.orchestrator import AgentOrchestrator
from ia_modules.agents.core import BaseAgent
from ia_modules.agents.roles import AgentRole
from ia_modules.agents.state import StateManager


class MultiAgentService:
    """Service for multi-agent workflow orchestration and visualization"""
    
    def __init__(self, storage_dir: str = "./workflows"):
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.websocket_callback: Optional[Callable] = None
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def set_websocket_callback(self, callback: Callable):
        """Set callback function for broadcasting WebSocket events"""
        self.websocket_callback = callback
        
    async def create_workflow(
        self,
        workflow_id: str,
        agents: List[Dict[str, Any]],
        edges: List[Dict[str, str]],
        feedback_loops: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a multi-agent workflow
        
        Args:
            workflow_id: Unique workflow identifier
            agents: List of agent configurations
            edges: List of connections between agents
            feedback_loops: Optional feedback loop configurations
            
        Returns:
            Workflow configuration and metadata
        """
        state = StateManager(thread_id=workflow_id)
        orchestrator = AgentOrchestrator(state)
        
        # Create and register agents
        agent_instances = {}
        for agent_config in agents:
            role = AgentRole(
                name=agent_config["role"],
                description=agent_config.get("description", "")
            )
            
            # Create specialized agent based on role
            agent = self._create_agent(role, state, agent_config)
            agent_instances[agent_config["id"]] = agent
            orchestrator.add_agent(agent_config["id"], agent)
        
        # Add edges
        for edge in edges:
            condition = None
            if "condition" in edge:
                condition = self._create_condition(edge["condition"])
            
            orchestrator.add_edge(
                edge["from"],
                edge["to"],
                condition=condition,
                metadata=edge.get("metadata", {})
            )
        
        # Add feedback loops
        if feedback_loops:
            for loop in feedback_loops:
                orchestrator.add_feedback_loop(
                    loop["from"],
                    loop["to"],
                    max_iterations=loop.get("max_iterations", 3)
                )
        
        # Store workflow
        workflow_data = {
            "id": workflow_id,
            "orchestrator": orchestrator,
            "state": state,
            "agents": agent_instances,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "created",
            "execution_log": []
        }
        
        self.active_workflows[workflow_id] = workflow_data
        
        return {
            "workflow_id": workflow_id,
            "num_agents": len(agents),
            "num_edges": len(edges),
            "visualization": orchestrator.visualize(),
            "stats": orchestrator.get_agent_stats()
        }
    
    async def execute_workflow(
        self,
        workflow_id: str,
        start_agent: str,
        initial_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a multi-agent workflow
        
        Args:
            workflow_id: Workflow to execute
            start_agent: Starting agent ID
            initial_data: Initial data to pass to workflow
            
        Returns:
            Execution results and communication log
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        orchestrator = workflow["orchestrator"]
        state = workflow["state"]
        
        # Track execution
        execution_id = f"{workflow_id}_{int(datetime.now(UTC).timestamp())}"
        workflow["status"] = "running"
        workflow["current_execution_id"] = execution_id
        
        # Execute workflow with communication tracking using hooks
        communication_log = []
        agent_stats = {}
        execution_path = []
        
        # Hook: Agent start
        async def on_agent_start(agent_id: str, input_data: Dict[str, Any]):
            start_time = datetime.now(UTC)
            event = {
                "timestamp": start_time.isoformat(),
                "type": "agent_activated",
                "agent": agent_id,
                "input_data": input_data
            }
            communication_log.append(event)
            execution_path.append(agent_id)
            
            # Broadcast via WebSocket
            if self.websocket_callback:
                await self.websocket_callback(workflow_id, "agent_start", event)
        
        # Hook: Agent complete
        async def on_agent_complete(agent_id: str, output_data: Dict[str, Any], duration: float):
            end_time = datetime.now(UTC)
            event = {
                "timestamp": end_time.isoformat(),
                "type": "agent_completed",
                "agent": agent_id,
                "output_data": output_data,
                "duration_seconds": duration
            }
            communication_log.append(event)
            
            # Track stats
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "executions": 0,
                    "total_duration": 0,
                    "iterations": []
                }
            
            agent_stats[agent_id]["executions"] += 1
            agent_stats[agent_id]["total_duration"] += duration
            agent_stats[agent_id]["iterations"].append({
                "iteration": agent_stats[agent_id]["executions"],
                "duration": duration,
                "timestamp": end_time.isoformat()
            })
            
            # Broadcast via WebSocket
            if self.websocket_callback:
                await self.websocket_callback(workflow_id, "agent_complete", event)
        
        # Hook: Agent error
        async def on_agent_error(agent_id: str, error: Exception):
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "agent_error",
                "agent": agent_id,
                "error": str(error)
            }
            communication_log.append(event)
            
            # Broadcast via WebSocket
            if self.websocket_callback:
                await self.websocket_callback(workflow_id, "agent_error", event)
        
        # Register hooks
        orchestrator.add_hook("agent_start", on_agent_start)
        orchestrator.add_hook("agent_complete", on_agent_complete)
        orchestrator.add_hook("agent_error", on_agent_error)
        
        try:
            # Run workflow
            result = await orchestrator.run(start_agent, initial_data)
            
            workflow["status"] = "completed"
            
            execution_record = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "start_agent": start_agent,
                "result": result,
                "communication_log": communication_log,
                "execution_path": execution_path,
                "agent_stats": agent_stats,
                "state_snapshot": await state.snapshot(),
                "completed_at": datetime.now(UTC).isoformat()
            }
            
            workflow["execution_log"].append(execution_record)
            self.execution_history.append(execution_record)
            
            return execution_record
            
        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            
            execution_record = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "start_agent": start_agent,
                "error": str(e),
                "communication_log": communication_log,
                "execution_path": execution_path,
                "failed_at": datetime.now(UTC).isoformat()
            }
            
            workflow["execution_log"].append(execution_record)
            self.execution_history.append(execution_record)
            
            raise
    
    async def get_workflow_state(self, workflow_id: str) -> Dict[str, Any]:
        """Get current state of a workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        state = workflow["state"]
        orchestrator = workflow["orchestrator"]
        
        return {
            "workflow_id": workflow_id,
            "status": workflow["status"],
            "state": await state.snapshot(),
            "stats": orchestrator.get_agent_stats(),
            "execution_log_count": len(workflow["execution_log"])
        }
    
    async def get_agent_communications(
        self,
        workflow_id: str,
        execution_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get agent communication log"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        if execution_id:
            # Get specific execution
            for execution in workflow["execution_log"]:
                if execution["execution_id"] == execution_id:
                    return execution["communication_log"]
            raise ValueError(f"Execution {execution_id} not found")
        else:
            # Get latest execution
            if workflow["execution_log"]:
                return workflow["execution_log"][-1]["communication_log"]
            return []
    
    def _create_agent(
        self,
        role: AgentRole,
        state: StateManager,
        config: Dict[str, Any]
    ) -> BaseAgent:
        """Create agent instance based on role"""
        
        # Create demo agents for different roles
        class DemoAgent(BaseAgent):
            """Demo agent for testing"""
            
            async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Execute agent task"""
                # Simulate agent processing
                await asyncio.sleep(0.1)  # Simulate work
                
                # Add role-specific behavior
                result = {
                    **data,
                    f"{self.role.name}_processed": True,
                    "processed_by": self.role.name,
                    "timestamp": datetime.now(UTC).isoformat()
                }
                
                # Role-specific modifications
                if "planner" in self.role.name.lower():
                    result["plan"] = ["Step 1", "Step 2", "Step 3"]
                elif "researcher" in self.role.name.lower():
                    result["research_data"] = ["Finding 1", "Finding 2"]
                elif "coder" in self.role.name.lower():
                    result["code"] = "def example(): pass"
                elif "critic" in self.role.name.lower():
                    result["feedback"] = "Looks good" if data.get("iteration", 0) > 1 else "Needs improvement"
                    result["approved"] = data.get("iteration", 0) > 1
                
                return result
        
        return DemoAgent(role, state)
    
    def _create_condition(self, condition_name: str) -> callable:
        """Create condition function from name"""
        
        def needs_revision(data: Dict[str, Any]) -> bool:
            return not data.get("approved", False)
        
        def is_complete(data: Dict[str, Any]) -> bool:
            return data.get("approved", True)
        
        def has_errors(data: Dict[str, Any]) -> bool:
            return "error" in data
        
        conditions = {
            "needs_revision": needs_revision,
            "is_complete": is_complete,
            "has_errors": has_errors
        }
        
        return conditions.get(condition_name, lambda x: True)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return [
            {
                "workflow_id": wf_id,
                "status": wf["status"],
                "num_agents": len(wf["agents"]),
                "created_at": wf["created_at"],
                "executions": len(wf["execution_log"])
            }
            for wf_id, wf in self.active_workflows.items()
        ]
    
    def get_execution_history(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get execution history across all workflows"""
        return self.execution_history[-limit:]
    
    async def save_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Save workflow configuration to persistent storage
        
        Args:
            workflow_id: Workflow ID to save
            name: Human-readable workflow name
            description: Optional workflow description
            
        Returns:
            Save confirmation with file path
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        orchestrator = workflow["orchestrator"]
        
        # Create serializable workflow configuration
        agents_config = []
        for agent_id, agent in workflow["agents"].items():
            agents_config.append({
                "id": agent_id,
                "role": agent.role.name,
                "description": agent.role.description
            })
        
        # Get edges from orchestrator
        edges_config = []
        for from_agent, edges in orchestrator.graph.items():
            for edge in edges:
                edge_dict = {
                    "from": from_agent,
                    "to": edge.to
                }
                if edge.condition:
                    edge_dict["condition"] = edge.condition.__name__ if callable(edge.condition) else str(edge.condition)
                edges_config.append(edge_dict)
        
        # Get feedback loops if any
        feedback_loops_config = []
        if hasattr(orchestrator, 'feedback_loops'):
            for loop_id, loop_data in orchestrator.feedback_loops.items():
                feedback_loops_config.append({
                    "from": loop_data.get("from"),
                    "to": loop_data.get("to"),
                    "max_iterations": loop_data.get("max_iterations", 3)
                })
        
        workflow_data = {
            "workflow_id": workflow_id,
            "name": name,
            "description": description,
            "agents": agents_config,
            "edges": edges_config,
            "feedback_loops": feedback_loops_config,
            "created_at": workflow["created_at"],
            "saved_at": datetime.now(UTC).isoformat(),
            "version": "1.0"
        }
        
        # Save to file
        file_path = self.storage_dir / f"{workflow_id}.json"
        with open(file_path, 'w') as f:
            json.dump(workflow_data, f, indent=2)
        
        return {
            "workflow_id": workflow_id,
            "name": name,
            "saved_at": workflow_data["saved_at"],
            "file_path": str(file_path)
        }
    
    async def load_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Load workflow configuration from persistent storage
        
        Args:
            workflow_id: Workflow ID to load
            
        Returns:
            Loaded workflow configuration
        """
        file_path = self.storage_dir / f"{workflow_id}.json"
        
        if not file_path.exists():
            raise ValueError(f"Workflow {workflow_id} not found in storage")
        
        with open(file_path, 'r') as f:
            workflow_data = json.load(f)
        
        # Recreate the workflow
        result = await self.create_workflow(
            workflow_id=workflow_data["workflow_id"],
            agents=workflow_data["agents"],
            edges=workflow_data["edges"],
            feedback_loops=workflow_data.get("feedback_loops")
        )
        
        # Update metadata
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["name"] = workflow_data.get("name")
            self.active_workflows[workflow_id]["description"] = workflow_data.get("description")
            self.active_workflows[workflow_id]["original_created_at"] = workflow_data.get("created_at")
        
        return {
            **result,
            "name": workflow_data.get("name"),
            "description": workflow_data.get("description"),
            "loaded_from": str(file_path)
        }
    
    async def list_saved_workflows(self) -> List[Dict[str, Any]]:
        """
        List all saved workflows
        
        Returns:
            List of saved workflow metadata
        """
        saved_workflows = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    workflow_data = json.load(f)
                
                saved_workflows.append({
                    "workflow_id": workflow_data["workflow_id"],
                    "name": workflow_data.get("name"),
                    "description": workflow_data.get("description"),
                    "num_agents": len(workflow_data.get("agents", [])),
                    "num_edges": len(workflow_data.get("edges", [])),
                    "created_at": workflow_data.get("created_at"),
                    "saved_at": workflow_data.get("saved_at"),
                    "file_path": str(file_path)
                })
            except Exception as e:
                print(f"Error loading workflow from {file_path}: {e}")
                continue
        
        return saved_workflows
    
    async def delete_saved_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Delete a saved workflow from storage
        
        Args:
            workflow_id: Workflow ID to delete
            
        Returns:
            Deletion confirmation
        """
        file_path = self.storage_dir / f"{workflow_id}.json"
        
        if not file_path.exists():
            raise ValueError(f"Workflow {workflow_id} not found in storage")
        
        file_path.unlink()
        
        return {
            "workflow_id": workflow_id,
            "deleted_at": datetime.now(UTC).isoformat(),
            "message": "Workflow deleted successfully"
        }

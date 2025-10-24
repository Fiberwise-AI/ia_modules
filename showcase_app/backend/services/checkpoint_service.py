"""Checkpoint service for managing pipeline execution checkpoints"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CheckpointService:
    """Service for checkpoint management using ia_modules"""

    def __init__(self, checkpointer=None, pipeline_service=None):
        """
        Initialize checkpoint service
        
        Args:
            checkpointer: Checkpointer instance from ia_modules (Redis or SQL)
            pipeline_service: PipelineService for resuming execution
        """
        self.checkpointer = checkpointer
        self.pipeline_service = pipeline_service
        logger.info(f"Checkpoint service initialized with {type(checkpointer).__name__ if checkpointer else 'no checkpointer'}")

    async def list_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a specific execution
        
        Args:
            job_id: Execution job ID
            
        Returns:
            List of checkpoint dictionaries
        """
        if not self.checkpointer:
            logger.warning("No checkpointer available")
            return []

        try:
            # Get checkpoints from checkpointer
            checkpoints = await self._get_checkpoints_from_storage(job_id)
            
            # Format for API response
            formatted = []
            for cp in checkpoints:
                formatted.append({
                    "id": cp.get("id") or cp.get("checkpoint_id"),
                    "job_id": job_id,
                    "step_name": cp.get("step_name"),
                    "created_at": cp.get("created_at"),
                    "state_size": len(str(cp.get("state", {}))) if cp.get("state") else 0,
                    "metadata": cp.get("metadata", {})
                })
            
            logger.info(f"Retrieved {len(formatted)} checkpoints for job {job_id}")
            return formatted

        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}", exc_info=True)
            return []

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific checkpoint by ID
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint dictionary or None
        """
        if not self.checkpointer:
            return None

        try:
            checkpoint = await self._load_checkpoint_from_storage(checkpoint_id)
            
            if not checkpoint:
                return None

            return {
                "id": checkpoint_id,
                "job_id": checkpoint.get("job_id"),
                "step_name": checkpoint.get("step_name"),
                "state": checkpoint.get("state", {}),
                "created_at": checkpoint.get("created_at"),
                "metadata": checkpoint.get("metadata", {})
            }

        except Exception as e:
            logger.error(f"Error getting checkpoint {checkpoint_id}: {e}", exc_info=True)
            return None

    async def get_checkpoint_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the state data from a checkpoint
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            State dictionary or None
        """
        checkpoint = await self.get_checkpoint(checkpoint_id)
        return checkpoint.get("state") if checkpoint else None

    async def resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Resume pipeline execution from a checkpoint
        
        Args:
            checkpoint_id: Checkpoint ID to resume from
            
        Returns:
            New execution details
        """
        if not self.checkpointer or not self.pipeline_service:
            raise ValueError("Checkpoint or pipeline service not available")

        try:
            # Load checkpoint
            checkpoint = await self.get_checkpoint(checkpoint_id)
            
            if not checkpoint:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            job_id = checkpoint["job_id"]
            state = checkpoint["state"]
            
            logger.info(f"Resuming execution {job_id} from checkpoint {checkpoint_id}")
            
            # Get original execution
            execution = await self.pipeline_service.get_execution(job_id)
            
            if not execution:
                raise ValueError(f"Original execution {job_id} not found")

            # Create new execution with checkpoint state
            # This is a simplified version - actual implementation depends on how
            # the pipeline runner handles checkpoint resumption
            new_job_id = await self.pipeline_service.execute_pipeline(
                pipeline_id=execution["pipeline_id"],
                input_data=state.get("input_data", execution.get("input_data", {})),
                checkpoint_enabled=True
            )

            return {
                "original_job_id": job_id,
                "new_job_id": new_job_id,
                "resumed_from_checkpoint": checkpoint_id,
                "resumed_at_step": checkpoint["step_name"]
            }

        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {e}", exc_info=True)
            raise

    async def _get_checkpoints_from_storage(self, job_id: str) -> List[Dict]:
        """Get checkpoints from storage backend"""
        # This is a placeholder - actual implementation depends on checkpointer interface
        try:
            if hasattr(self.checkpointer, 'list_checkpoints'):
                return await self.checkpointer.list_checkpoints(job_id)
            elif hasattr(self.checkpointer, 'get_checkpoints'):
                return await self.checkpointer.get_checkpoints(job_id)
            else:
                # Fallback: try to get from database directly
                if hasattr(self.checkpointer, 'db_manager'):
                    query = """
                        SELECT checkpoint_id as id, job_id, step_name, 
                               state, created_at, metadata
                        FROM checkpoints
                        WHERE job_id = :job_id
                        ORDER BY created_at DESC
                    """
                    result = await self.checkpointer.db_manager.execute(
                        query,
                        {"job_id": job_id}
                    )
                    return [dict(row) for row in result]
                
                return []
        except Exception as e:
            logger.error(f"Error accessing checkpoint storage: {e}")
            return []

    async def _load_checkpoint_from_storage(self, checkpoint_id: str) -> Optional[Dict]:
        """Load checkpoint from storage backend"""
        try:
            if hasattr(self.checkpointer, 'load_checkpoint'):
                cp = await self.checkpointer.load_checkpoint(checkpoint_id)
                if cp:
                    return self._checkpoint_to_dict(cp)
            elif hasattr(self.checkpointer, 'get_checkpoint'):
                cp = await self.checkpointer.get_checkpoint(checkpoint_id)
                if cp:
                    return self._checkpoint_to_dict(cp)
            elif hasattr(self.checkpointer, 'db_manager'):
                query = """
                    SELECT checkpoint_id as id, job_id, step_name,
                           state, created_at, metadata
                    FROM checkpoints
                    WHERE checkpoint_id = :checkpoint_id
                """
                result = await self.checkpointer.db_manager.execute(
                    query,
                    {"checkpoint_id": checkpoint_id}
                )
                rows = list(result)
                if rows:
                    return dict(rows[0])
            
            return None
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def _checkpoint_to_dict(self, checkpoint) -> Dict:
        """Convert checkpoint object to dictionary"""
        if isinstance(checkpoint, dict):
            return checkpoint

        return {
            "id": getattr(checkpoint, "id", None) or getattr(checkpoint, "checkpoint_id", None),
            "job_id": getattr(checkpoint, "job_id", None),
            "step_name": getattr(checkpoint, "step_name", None),
            "state": getattr(checkpoint, "state", {}),
            "created_at": getattr(checkpoint, "created_at", None),
            "metadata": getattr(checkpoint, "metadata", {})
        }

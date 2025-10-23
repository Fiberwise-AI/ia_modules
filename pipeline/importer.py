"""
Shared Pipeline Importer

Moved from the app-level implementation so that multiple consuming
applications can reuse the same importer logic from `ia_modules`.

The constructor accepts an optional `pipelines_dir`. If not provided
the importer will look for a `pipelines/` directory in the current
working directory. Consuming apps should pass an explicit path if
they keep pipelines adjacent to their application package.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

from ia_modules.database.interfaces import DatabaseInterface

logger = logging.getLogger(__name__)


class PipelineImportService:
    """Service for importing pipeline definitions from JSON files"""

    def __init__(self, db_provider: DatabaseInterface, pipelines_dir: Optional[str] = None):
        self.db_provider = db_provider
        if pipelines_dir is None:
            # Default to a pipelines folder in the current working directory
            self.pipelines_dir = Path.cwd() / "pipelines"
        else:
            self.pipelines_dir = Path(pipelines_dir)

    async def clear_all_imported_pipelines(self) -> int:
        """Clear all imported pipelines (non-system) from database"""
        logger.info("Clearing all imported pipelines from database")

        query = "DELETE FROM pipelines WHERE is_system = 0"
        result = await self.db_provider.execute_async(query)

        if not result.success:
            logger.error(f"Failed to clear pipelines: {result}")
            return 0

        # Get count of deleted rows (if available)
        deleted_count = getattr(result, 'rows_affected', 0)
        logger.info(f"Cleared {deleted_count} imported pipelines")
        return deleted_count

    async def import_all_pipelines(self, clear_existing: bool = False) -> Dict[str, Any]:
        """Import all pipeline JSON files from the pipelines directory"""
        logger.info(f"Starting pipeline import from: {self.pipelines_dir}")

        # Clear existing pipelines if requested
        if clear_existing:
            await self.clear_all_imported_pipelines()

        if not self.pipelines_dir.exists():
            logger.warning(f"Pipelines directory does not exist: {self.pipelines_dir}")
            return {"imported": 0, "skipped": 0, "errors": 0}

        json_files = list(self.pipelines_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")

        results = {
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }

        for json_file in json_files:
            try:
                result = await self._import_pipeline_file(json_file)
                results[result["action"]] += 1
                results["details"].append({
                    "file": json_file.name,
                    "action": result["action"],
                    "pipeline_id": result.get("pipeline_id"),
                    "slug": result.get("slug")
                })
            except Exception as e:
                logger.error(f"Failed to import {json_file.name}: {e}")
                results["errors"] += 1
                results["details"].append({
                    "file": json_file.name,
                    "action": "error",
                    "error": str(e)
                })

        logger.info(f"Pipeline import completed: {results['imported']} imported, {results['updated']} updated, {results['skipped']} skipped, {results['errors']} errors")
        return results

    async def _import_pipeline_file(self, file_path: Path) -> Dict[str, Any]:
        """Import a single pipeline JSON file"""
        logger.debug(f"Processing pipeline file: {file_path.name}")

        # Load and validate JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        # Validate required fields
        if not isinstance(pipeline_data, dict):
            raise ValueError("Pipeline file must contain a JSON object")

        name = pipeline_data.get('name')
        if not name:
            raise ValueError("Pipeline must have a 'name' field")

        # Generate slug from filename (without extension)
        slug = file_path.stem

        # Calculate content hash for change detection
        content_hash = self._calculate_content_hash(pipeline_data)

        # Check if pipeline already exists
        existing = await self._get_existing_pipeline(slug)

        if existing:
            # Check if content has changed
            if existing['content_hash'] == content_hash:
                logger.debug(f"Pipeline {slug} unchanged, skipping")
                return {"action": "skipped", "pipeline_id": existing['id'], "slug": slug}

            # Update existing pipeline
            await self._update_pipeline(existing['id'], pipeline_data, content_hash, str(file_path.relative_to(self.pipelines_dir)))
            logger.info(f"Updated pipeline: {slug}")
            return {"action": "updated", "pipeline_id": existing['id'], "slug": slug}

        # Create new pipeline
        pipeline_id = await self._create_pipeline(slug, pipeline_data, content_hash, str(file_path.relative_to(self.pipelines_dir)))
        logger.info(f"Imported new pipeline: {slug} ({pipeline_id})")
        return {"action": "imported", "pipeline_id": pipeline_id, "slug": slug}

    def _generate_slug(self, name: str, file_path: str) -> str:
        """Generate a unique slug from name and file path"""
        clean_name = name.lower().replace(' ', '-').replace('_', '-')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '-')
        while '--' in clean_name:
            clean_name = clean_name.replace('--', '-')
        clean_name = clean_name.strip('-')

        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:6]
        slug = f"{clean_name}-{path_hash}"

        return slug

    def _calculate_content_hash(self, pipeline_data: Dict[str, Any]) -> str:
        """Calculate hash of pipeline content for change detection"""
        content_str = json.dumps(pipeline_data, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    async def _get_existing_pipeline(self, slug: str) -> Optional[Dict[str, Any]]:
        """Check if pipeline already exists by slug"""
        query = """
        SELECT id, content_hash FROM pipelines
        WHERE slug = ? AND is_system = 0
        """

        result = await self.db_provider.fetch_one(query, (slug,))
        if result and result.data:
            row = result.data[0] if isinstance(result.data, list) else result.data
            return dict(row)
        return None

    async def _create_pipeline(self, slug: str, pipeline_data: Dict[str, Any], content_hash: str, file_path: str) -> str:
        """Create a new pipeline in the database"""
        import uuid
        pipeline_id = str(uuid.uuid4())

        pipeline_json = json.dumps(pipeline_data)
        name = pipeline_data.get('name', 'Unnamed Pipeline')
        description = pipeline_data.get('description', '')
        version = pipeline_data.get('version', '1.0')

        query = """
        INSERT INTO pipelines (
            id, slug, name, description, version, pipeline_json,
            file_path, content_hash, is_system
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """

        result = await self.db_provider.execute_async(
            query,
            (pipeline_id, slug, name, description, version, pipeline_json, file_path, content_hash)
        )

        if not result.success:
            raise Exception(f"Failed to create pipeline: {result}")

        return pipeline_id

    async def _update_pipeline(self, pipeline_id: str, pipeline_data: Dict[str, Any], content_hash: str, file_path: str):
        """Update an existing pipeline"""
        pipeline_json = json.dumps(pipeline_data)
        name = pipeline_data.get('name', 'Unnamed Pipeline')
        description = pipeline_data.get('description', '')
        version = pipeline_data.get('version', '1.0')

        query = """
        UPDATE pipelines
        SET name = ?, description = ?, version = ?, pipeline_json = ?,
            file_path = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """

        result = await self.db_provider.execute_async(
            query,
            (name, description, version, pipeline_json, file_path, content_hash, pipeline_id)
        )

        if not result.success:
            raise Exception(f"Failed to update pipeline: {result}")

    async def get_pipeline_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get pipeline by slug"""
        query = """
        SELECT id, slug, name, description, version, pipeline_json,
               file_path, is_active, created_at, updated_at
        FROM pipelines
        WHERE slug = ? AND is_active = 1
        """

        result = await self.db_provider.fetch_one(query, (slug,))
        if result and result.data:
            row = result.data[0] if isinstance(result.data, list) else result.data
            pipeline_data = dict(row)
            pipeline_data['pipeline_config'] = json.loads(pipeline_data['pipeline_json'])
            return pipeline_data
        return None

    async def list_imported_pipelines(self) -> List[Dict[str, Any]]:
        """List all imported pipelines"""
        query = """
        SELECT id, slug, name, description, version, file_path,
               created_at, updated_at
        FROM pipelines
        WHERE is_system = 0 AND is_active = 1
        ORDER BY name
        """

        result = await self.db_provider.fetch_all(query)
        if result and result.success and result.data:
            return [dict(row) for row in result.data]
        return []

    async def validate_pipeline_config(self, pipeline_data: Dict[str, Any]) -> bool:
        """Validate pipeline configuration structure"""
        try:
            required_fields = ['name', 'steps']
            for field in required_fields:
                if field not in pipeline_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            steps = pipeline_data.get('steps', [])
            if not isinstance(steps, list) or len(steps) == 0:
                logger.error("Pipeline must have at least one step")
                return False

            for i, step in enumerate(steps):
                required_step_fields = ['id', 'step_class', 'module']
                for field in required_step_fields:
                    if field not in step:
                        logger.error(f"Step {i} missing required field: {field}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Pipeline validation error: {e}")
            return False

"""
Data Ingestor Step Implementation
"""

from typing import Dict, Any
import json

from ia_modules.pipeline.core import Step


class DataIngestorStep(Step):
    """Step to ingest data from a source"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data from the configured source or use input data"""
        # First try to get test data from config
        test_data = self.config.get('test_data')

        # If no test data in config, use the input data directly
        if not test_data:
            raw_data = data.get('raw_data', data)
        else:
            raw_data = test_data

        return {
            "ingested_data": raw_data,
            "source": "inline_test_data",
            "record_count": len(raw_data) if isinstance(raw_data, list) else 1
        }

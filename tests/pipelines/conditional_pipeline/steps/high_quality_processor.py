"""
High Quality Processor Step Implementation
"""

from typing import Dict, Any

from ia_modules.pipeline.core import Step


class HighQualityProcessorStep(Step):
    """Step to process high quality data with full processing"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process high quality data with full processing"""
        raw_data = data.get('ingested_data', [])
        quality_score = data.get('quality_score', 0)
        
        # Full processing for high quality data
        processed_data = self._process_full(raw_data)
        
        return {
            "processed_data": processed_data,
            "processing_level": "full",
            "quality_score": quality_score,
            "processing_info": {
                "records_processed": len(processed_data) if isinstance(processed_data, list) else 1
            }
        }
        
    def _process_full(self, data: Any) -> Any:
        """Perform full processing on the data"""
        # This is a simplified example - in practice this would be more complex
        if isinstance(data, dict):
            return self._process_record_full(data)
        elif isinstance(data, list):
            return [self._process_record_full(record) for record in data]
        else:
            return data
            
    def _process_record_full(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record with full processing"""
        # Add some transformations
        processed = record.copy()
        
        # Example transformation - normalize text fields
        for key, value in processed.items():
            if isinstance(value, str):
                processed[key] = value.strip().lower()
                
        return processed

"""
Quality Checker Step Implementation
"""

from typing import Dict, Any

from ia_modules.pipeline.core import Step


class QualityCheckerStep(Step):
    """Step to check the quality of data"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data quality and return quality score"""
        raw_data = data.get('ingested_data', [])
        
        if not isinstance(raw_data, (list, dict)):
            raise ValueError("Data must be a list or dictionary for quality checking")
            
        # Simple quality check - count non-null fields
        quality_score = self._calculate_quality_score(raw_data)
        
        return {
            "ingested_data": raw_data,  # Pass through the data
            "quality_score": quality_score,
            "data_quality": self._get_quality_description(quality_score),
            "quality_details": {
                "total_records": len(raw_data) if isinstance(raw_data, list) else 1,
                "score": quality_score
            }
        }
        
    def _calculate_quality_score(self, data: Any) -> float:
        """Calculate a simple quality score"""
        if isinstance(data, dict):
            # For single record, check how many fields are not null
            non_null_fields = sum(1 for value in data.values() if value is not None)
            total_fields = len(data)
            return non_null_fields / total_fields if total_fields > 0 else 0
            
        elif isinstance(data, list) and len(data) > 0:
            # For multiple records, calculate average quality
            scores = []
            for record in data:
                if isinstance(record, dict):
                    non_null_fields = sum(1 for value in record.values() if value is not None)
                    total_fields = len(record)
                    score = non_null_fields / total_fields if total_fields > 0 else 0
                    scores.append(score)
            
            return sum(scores) / len(scores) if scores else 0
            
        else:
            return 0.5  # Default quality for unknown data types
            
    def _get_quality_description(self, score: float) -> str:
        """Get a human-readable quality description"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Poor"

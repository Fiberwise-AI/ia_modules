"""
Results Merger Step Implementation
"""

from typing import Dict, Any, List

from ia_modules.pipeline.core import Step


class ResultsMergerStep(Step):
    """Step to merge results from parallel processing streams"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from different parallel processing streams"""
        # Get all the processed data from different streams using new input format
        stream_results = []

        # Collect results from all parallel processors
        processed_data_1 = input.get('processed_data_1', {})
        processed_data_2 = input.get('processed_data_2', {})
        processed_data_3 = input.get('processed_data_3', {})

        # Add non-empty results to the list
        for data in [processed_data_1, processed_data_2, processed_data_3]:
            if data:
                stream_results.append(data)
                
        # If we don't find stream-specific data, try to merge what's available
        if not stream_results:
            # Try to extract any data that looks like processed results
            for key, value in data.items():
                if isinstance(value, (list, dict)) and key != 'data_chunks':
                    stream_results.append(value)
        
        # Merge the results
        merged_data = self._merge_results(stream_results)
        
        return {
            "merged_results": merged_data,
            "stream_count": len(stream_results),
            "total_records": len(merged_data) if isinstance(merged_data, list) else 1,
            "processing_summary": self._get_processing_summary(stream_results)
        }
        
    def _merge_results(self, results: List[Any]) -> Any:
        """Merge multiple result sets"""
        # Simple merge - concatenate lists or combine dictionaries
        if not results:
            return {}
            
        # If all results are lists, concatenate them
        if all(isinstance(r, list) for r in results):
            merged = []
            for result in results:
                if isinstance(result, list):
                    merged.extend(result)
                else:
                    merged.append(result)
            return merged
            
        # If all results are dictionaries, merge them
        elif all(isinstance(r, dict) for r in results):
            merged = {}
            for result in results:
                if isinstance(result, dict):
                    merged.update(result)
            return merged
            
        # Otherwise, just return the first result
        else:
            return results[0] if results else {}
            
    def _get_processing_summary(self, results: List[Any]) -> Dict[str, Any]:
        """Get a summary of the merging process"""
        total_records = 0
        for result in results:
            if isinstance(result, list):
                total_records += len(result)
            elif isinstance(result, dict):
                total_records += 1
                
        return {
            "total_streams": len(results),
            "total_merged_records": total_records,
            "merge_type": "concatenation" if all(isinstance(r, list) for r in results) else "combination"
        }

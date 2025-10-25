"""
Test to verify parallel pipeline works correctly in showcase app context
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directories to path
current_dir = Path(__file__).parent
showcase_dir = current_dir.parent
ia_modules_dir = showcase_dir.parent
sys.path.insert(0, str(ia_modules_dir))
sys.path.insert(0, str(showcase_dir))

from ia_modules.pipeline.runner import create_pipeline_from_json
from tests.pipeline_runner import run_with_new_schema


async def test_parallel_pipeline_output():
    """Test that parallel pipeline produces non-empty statistics"""
    
    # Load the parallel pipeline
    pipeline_file = ia_modules_dir / "tests" / "pipelines" / "parallel_pipeline" / "pipeline.json"
    
    print(f"Loading pipeline from: {pipeline_file}")
    
    with open(pipeline_file, 'r') as f:
        pipeline_config = json.load(f)
    
    # Test data similar to what showcase app would use
    input_data = {
        "loaded_data": [
            {"id": 1, "value": 100, "category": "A"},
            {"id": 2, "value": 200, "category": "B"},
            {"id": 3, "value": 300, "category": "C"},
            {"id": 4, "value": 150, "category": "A"},
            {"id": 5, "value": 250, "category": "B"},
            {"id": 6, "value": 350, "category": "C"},
        ]
    }
    
    print(f"\nInput data: {len(input_data['loaded_data'])} items")
    
    # Create and run pipeline
    pipeline = create_pipeline_from_json(pipeline_config)
    result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION RESULTS")
    print("="*60)
    
    # Check step results
    for step_id in ["step1", "step2", "step3", "step4", "step5", "step6"]:
        if step_id in result["steps"]:
            step_result = result["steps"][step_id]
            print(f"\n{step_id.upper()}:")
            
            if step_id == "step1":  # Data Splitter
                print(f"  - Chunks: {step_result.get('chunk_count', 0)}")
                print(f"  - Original size: {step_result.get('original_data_size', 0)}")
                
            elif step_id in ["step2", "step3", "step4"]:  # Parallel Processors
                print(f"  - Stream ID: {step_result.get('stream_id', 'N/A')}")
                print(f"  - Records: {step_result.get('record_count', 0)}")
                
            elif step_id == "step5":  # Results Merger
                print(f"  - Stream count: {step_result.get('stream_count', 0)}")
                print(f"  - Total records: {step_result.get('total_records', 0)}")
                merged_results = step_result.get('merged_results', {})
                if isinstance(merged_results, dict) and 'processed_data' in merged_results:
                    print(f"  - Merged data length: {len(merged_results['processed_data'])}")
                
            elif step_id == "step6":  # Stats Collector (FINAL OUTPUT)
                print(f"  - Total streams: {step_result.get('total_streams', 0)}")
                print(f"  - Total records processed: {step_result.get('total_records_processed', 0)}")
                statistics = step_result.get('statistics', [])
                print(f"  - Statistics entries: {len(statistics)}")
                
                # Show detailed statistics
                if statistics:
                    print("\n  Statistics breakdown:")
                    for stat in statistics:
                        print(f"    Stream {stat.get('stream_id')}: "
                              f"{stat.get('records_processed')} records")
    
    # Verify final output is NOT empty
    step6_result = result["steps"]["step6"]
    
    print("\n" + "="*60)
    print("FINAL OUTPUT VERIFICATION")
    print("="*60)
    
    total_records = step6_result.get('total_records_processed', 0)
    total_streams = step6_result.get('total_streams', 0)
    statistics = step6_result.get('statistics', [])
    
    success = True
    
    if total_records == 0:
        print("❌ FAIL: total_records_processed is 0 (should be > 0)")
        success = False
    else:
        print(f"✅ PASS: total_records_processed = {total_records}")
    
    if total_streams == 0:
        print("❌ FAIL: total_streams is 0 (should be > 0)")
        success = False
    else:
        print(f"✅ PASS: total_streams = {total_streams}")
    
    if len(statistics) == 0:
        print("❌ FAIL: statistics array is empty")
        success = False
    else:
        print(f"✅ PASS: statistics has {len(statistics)} entries")
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL CHECKS PASSED - Parallel pipeline produces correct output!")
    else:
        print("❌ SOME CHECKS FAILED - Review the output above")
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(test_parallel_pipeline_output())
    sys.exit(0 if success else 1)

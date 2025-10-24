"""
Quick test to verify LLM fallback logic works correctly
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from services.pattern_service import PatternService


async def test_reflection_without_llm():
    """Test reflection pattern works without LLM configured"""
    print("🧪 Testing Reflection Pattern (without LLM)...")
    
    service = PatternService()
    
    # Verify LLM service is None (no API keys)
    assert service.llm_service is None, "Expected llm_service to be None when no API keys configured"
    print("✅ LLM service is None as expected")
    
    # Test reflection pattern - should fall back to simulated responses
    result = await service.reflection_example(
        initial_output="This is a test output that could be improved.",
        criteria={
            "clarity": "Text should be clear and understandable",
            "accuracy": "Information should be precise"
        },
        max_iterations=2
    )
    
    print(f"✅ Reflection completed with {result['total_iterations']} iterations")
    print(f"   Final quality score: {result['final_quality_score']:.2f}")
    assert result['pattern'] == 'reflection'
    assert 'iterations' in result
    

async def test_planning_without_llm():
    """Test planning pattern works without LLM configured"""
    print("\n🧪 Testing Planning Pattern (without LLM)...")
    
    service = PatternService()
    
    result = await service.planning_example(
        goal="Build a simple web application",
        constraints={"time": "2 weeks", "budget": "$1000"}
    )
    
    print(f"✅ Planning completed with {result['total_steps']} steps")
    assert result['pattern'] == 'planning'
    assert len(result['plan']) > 0
    

async def test_tool_use_without_llm():
    """Test tool use pattern works without LLM configured"""
    print("\n🧪 Testing Tool Use Pattern (without LLM)...")
    
    service = PatternService()
    
    result = await service.tool_use_example(
        task="Search for information and create a report",
        available_tools=["web_search", "document_writer", "calculator"]
    )
    
    print(f"✅ Tool use completed, selected {len(result.get('selected_tools', []))} tools")
    assert result['pattern'] == 'tool_use'
    

async def test_rag_without_llm():
    """Test RAG pattern works without LLM configured"""
    print("\n🧪 Testing RAG Pattern (without LLM)...")
    
    service = PatternService()
    
    result = await service.agentic_rag_example(
        initial_query="What is machine learning?",
        max_refinements=2
    )
    
    print(f"✅ RAG completed with {result['total_iterations']} iterations")
    print(f"   Final relevance: {result['final_relevance']:.2f}")
    assert result['pattern'] == 'agentic_rag'
    

async def test_metacognition_without_llm():
    """Test metacognition pattern works without LLM configured"""
    print("\n🧪 Testing Metacognition Pattern (without LLM)...")
    
    service = PatternService()
    
    result = await service.metacognition_example(
        execution_trace=[
            {"action": "retrieve", "duration": 0.5},
            {"action": "process", "duration": 1.2},
            {"action": "respond", "duration": 0.3}
        ],
        performance_metrics={
            "speed": 0.8,
            "accuracy": 0.9,
            "efficiency": 0.7
        }
    )
    
    print(f"✅ Metacognition completed")
    print(f"   Overall score: {result['analysis']['assessment']['overall_score']:.2f}")
    assert result['pattern'] == 'metacognition'
    

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Pattern Service LLM Fallback Logic")
    print("=" * 60)
    
    try:
        await test_reflection_without_llm()
        await test_planning_without_llm()
        await test_tool_use_without_llm()
        await test_rag_without_llm()
        await test_metacognition_without_llm()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n💡 Pattern service works correctly without LLM configuration.")
        print("   All patterns fall back to simulated responses gracefully.")
        print("\n📝 Next steps:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your API key (OpenAI, Anthropic, or Gemini)")
        print("   3. Test again to see real LLM responses")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

import React, { useState } from 'react';
import { Brain, Target, Wrench, Search, Activity, Play, Sparkles } from 'lucide-react';
import ReflectionViz from '../components/patterns/ReflectionViz';
import PlanningViz from '../components/patterns/PlanningViz';
import AgenticRAGViz from '../components/patterns/AgenticRAGViz';

/**
 * Agentic Patterns Page
 * Demonstrates advanced agentic design patterns
 */
export default function PatternsPage() {
  const [selectedPattern, setSelectedPattern] = useState('reflection');
  const [patternData, setPatternData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const patterns = [
    {
      id: 'reflection',
      name: 'Reflection',
      icon: Brain,
      color: 'purple',
      description: 'Self-critique and iterative improvement',
      example: {
        initial_output: 'AI is useful.',
        criteria: {
          clarity: 'Explain clearly and concisely',
          completeness: 'Cover key aspects comprehensively',
          accuracy: 'Provide factual information'
        }
      }
    },
    {
      id: 'planning',
      name: 'Planning',
      icon: Target,
      color: 'blue',
      description: 'Multi-step goal decomposition',
      example: {
        goal: 'Research the impact of AI on education',
        constraints: {
          time: '2 hours',
          depth: 'comprehensive'
        }
      }
    },
    {
      id: 'tool-use',
      name: 'Tool Use',
      icon: Wrench,
      color: 'orange',
      description: 'Dynamic tool selection',
      example: {
        task: 'Find and analyze recent research papers on quantum computing',
        available_tools: ['web_search', 'database_query', 'llm_analyzer', 'python_executor', 'file_system']
      }
    },
    {
      id: 'agentic-rag',
      name: 'Agentic RAG',
      icon: Search,
      color: 'green',
      description: 'Query refinement and retrieval',
      example: {
        query: 'machine learning applications',
        max_refinements: 3
      }
    },
    {
      id: 'metacognition',
      name: 'Metacognition',
      icon: Activity,
      color: 'pink',
      description: 'Self-monitoring and adaptation',
      example: {
        execution_trace: [
          { step: 'search', status: 'success', duration: 2.3 },
          { step: 'analyze', status: 'success', duration: 1.8 },
          { step: 'generate', status: 'error', duration: 0.5 },
          { step: 'retry_generate', status: 'success', duration: 2.1 }
        ],
        performance_metrics: {
          accuracy: 0.85,
          efficiency: 0.72,
          reliability: 0.90
        }
      }
    }
  ];

  const currentPattern = patterns.find(p => p.id === selectedPattern);

  const runPattern = async () => {
    setIsLoading(true);
    try {
      const endpoint = `/api/patterns/${selectedPattern}`;
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentPattern.example)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPatternData(data);
    } catch (error) {
      console.error('Error running pattern:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-4 mb-8">
        <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-lg">
          <Sparkles className="text-white" size={32} />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Agentic Design Patterns</h1>
          <p className="text-gray-600 mt-1">
            Explore advanced patterns for building intelligent, self-improving agents
          </p>
        </div>
      </div>

      {/* Pattern Selection */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Select a Pattern</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {patterns.map((pattern) => {
            const Icon = pattern.icon;
            const isSelected = selectedPattern === pattern.id;
            
            const colorClasses = {
              purple: 'from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700',
              blue: 'from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700',
              orange: 'from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700',
              green: 'from-green-500 to-green-600 hover:from-green-600 hover:to-green-700',
              pink: 'from-pink-500 to-pink-600 hover:from-pink-600 hover:to-pink-700'
            };

            return (
              <button
                key={pattern.id}
                onClick={() => {
                  setSelectedPattern(pattern.id);
                  setPatternData(null);
                }}
                className={`
                  p-4 rounded-lg border-2 transition-all duration-200
                  ${isSelected 
                    ? 'border-gray-800 shadow-lg scale-105' 
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                  }
                `}
              >
                <div className={`
                  w-12 h-12 mx-auto mb-3 rounded-lg flex items-center justify-center
                  bg-gradient-to-br ${colorClasses[pattern.color]} shadow-md
                `}>
                  <Icon className="text-white" size={24} />
                </div>
                <div className="font-semibold text-gray-800 text-sm mb-1">
                  {pattern.name}
                </div>
                <div className="text-xs text-gray-600">
                  {pattern.description}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Pattern Configuration */}
      {currentPattern && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">Pattern Configuration</h2>
            <button
              onClick={runPattern}
              disabled={isLoading}
              className={`
                px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-all
                ${isLoading 
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 shadow-md hover:shadow-lg'
                }
              `}
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                  <span>Running...</span>
                </>
              ) : (
                <>
                  <Play size={16} />
                  <span>Run Pattern</span>
                </>
              )}
            </button>
          </div>

          {/* Example Configuration Display */}
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="text-sm font-medium text-gray-700 mb-2">Example Configuration:</div>
            <pre className="text-xs text-gray-600 overflow-x-auto">
              {JSON.stringify(currentPattern.example, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* Pattern Visualization */}
      {patternData && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-6">Pattern Execution Results</h2>
          
          {selectedPattern === 'reflection' && <ReflectionViz data={patternData} />}
          {selectedPattern === 'planning' && <PlanningViz data={patternData} />}
          {selectedPattern === 'agentic-rag' && <AgenticRAGViz data={patternData} />}
          
          {selectedPattern === 'tool-use' && (
            <div className="space-y-4">
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <div className="font-semibold text-gray-800 mb-2">Task Analysis</div>
                <div className="text-sm text-gray-700 mb-3">{patternData.task}</div>
                <div className="flex flex-wrap gap-2">
                  {patternData.requirements?.map((req, idx) => (
                    <span key={idx} className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-xs font-medium">
                      {req}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <div className="font-semibold text-gray-800 mb-3">Tool Usage Plan</div>
                {patternData.usage_plan?.map((step, idx) => (
                  <div key={idx} className="mb-3 bg-white border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold text-sm">
                        {step.step}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-gray-800 mb-1">{step.tool}</div>
                        <div className="text-sm text-gray-600 mb-2">{step.reasoning}</div>
                        <div className="text-xs text-gray-500">
                          <span className="font-medium">Input:</span> {step.input} → 
                          <span className="font-medium ml-2">Output:</span> {step.expected_output}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {selectedPattern === 'metacognition' && (
            <div className="space-y-4">
              <div className="bg-pink-50 p-4 rounded-lg border border-pink-200">
                <div className="font-semibold text-gray-800 mb-3">Performance Assessment</div>
                <div className="text-lg font-bold text-pink-600 mb-2">
                  Overall Score: {(patternData.performance_assessment?.overall_score * 100).toFixed(0)}%
                </div>
                <div className="grid md:grid-cols-2 gap-4 mt-4">
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-2">Strengths</div>
                    <ul className="space-y-1">
                      {patternData.performance_assessment?.strengths?.map((s, idx) => (
                        <li key={idx} className="text-sm text-green-700">✓ {s}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-2">Weaknesses</div>
                    <ul className="space-y-1">
                      {patternData.performance_assessment?.weaknesses?.map((w, idx) => (
                        <li key={idx} className="text-sm text-red-700">✗ {w}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              
              {patternData.strategy_adjustments?.length > 0 && (
                <div>
                  <div className="font-semibold text-gray-800 mb-3">Strategy Adjustments</div>
                  {patternData.strategy_adjustments.map((adj, idx) => (
                    <div key={idx} className="mb-3 bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <div className="font-medium text-gray-800 mb-1">
                        {adj.aspect.charAt(0).toUpperCase() + adj.aspect.slice(1)}
                      </div>
                      <div className="text-sm text-gray-700 mb-1">{adj.suggestion}</div>
                      <div className="text-xs text-blue-600 font-medium">
                        Expected Impact: {adj.expected_impact}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Information Footer */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200 p-6">
        <h3 className="font-semibold text-gray-800 mb-3">About Agentic Patterns</h3>
        <p className="text-sm text-gray-700 mb-4">
          These patterns demonstrate the fundamental building blocks of AI agents. Each pattern shows
          how simple LLM text generation evolves into intelligent, autonomous behavior through the addition
          of system prompts, tools, memory, and reasoning patterns. This is what frameworks like LangChain
          and CrewAI implement under the hood.
        </p>
        
        <div className="bg-white rounded-lg p-4 mb-4 border border-blue-200">
          <div className="text-sm font-mono text-gray-700">
            <div className="font-semibold text-blue-600 mb-2">Core Agent Architecture:</div>
            <div className="text-xs">
              AI Agent = LLM + System Prompt + Tools + Memory + Reasoning Pattern<br/>
              <span className="text-gray-500">
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;─┬─&nbsp;&nbsp;&nbsp;──────┬──────&nbsp;&nbsp;&nbsp;──┬──&nbsp;&nbsp;&nbsp;──┬───&nbsp;&nbsp;&nbsp;────────┬────────<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Brain&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Identity&nbsp;&nbsp;&nbsp;&nbsp;Hands&nbsp;&nbsp;&nbsp;State&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Strategy
              </span>
            </div>
          </div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div>
            <span className="font-medium text-gray-800">Evolution:</span> Basic LLM → Specialized → 
            Tool-Using → Memory → ReAct → Metacognitive
          </div>
          <div>
            <span className="font-medium text-gray-800">Key Insight:</span> Frameworks abstract these patterns. 
            Understanding them deeply helps you use frameworks wisely.
          </div>
        </div>
      </div>
    </div>
  );
}

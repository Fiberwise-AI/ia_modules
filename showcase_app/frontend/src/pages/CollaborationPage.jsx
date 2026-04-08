import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { collaborationAPI } from '../services/api'
import toast from 'react-hot-toast'
import {
  Users,
  MessageSquare,
  GitBranch,
  Play,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  XCircle,
  Clock,
  ArrowRight,
  CheckCircle2,
  Shield,
  Share2,
} from 'lucide-react'

const PATTERN_ICONS = {
  consensus: <CheckCircle2 size={28} />,
  debate: <Shield size={28} />,
  hierarchical: <GitBranch size={28} />,
  peer_to_peer: <Share2 size={28} />,
}

const PATTERN_COLORS = {
  consensus: 'from-emerald-500 to-teal-600',
  debate: 'from-red-500 to-orange-600',
  hierarchical: 'from-blue-500 to-indigo-600',
  peer_to_peer: 'from-purple-500 to-pink-600',
}

const PATTERN_BG = {
  consensus: 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-800',
  debate: 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800',
  hierarchical: 'bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800',
  peer_to_peer: 'bg-purple-50 dark:bg-purple-950/30 border-purple-200 dark:border-purple-800',
}

const DEFAULT_CONFIGS = {
  consensus: {
    topic: 'We should adopt a microservices architecture for our next project',
    agents: ['Analyst', 'Engineer', 'Designer', 'PM', 'QA'],
    strategy: 'majority',
    max_iterations: 3,
  },
  debate: {
    topic: 'AI systems should be open source by default',
    proponents: ['Advocate-Open', 'Advocate-Community'],
    opponents: ['Skeptic-Security', 'Skeptic-Business'],
    moderator: 'Moderator',
    rounds: 2,
  },
  hierarchical: {
    task: 'Analyze market trends in the AI sector and produce a summary report',
    leader: 'Project-Lead',
    workers: ['Researcher-Alpha', 'Analyst-Beta', 'Writer-Gamma'],
  },
  peer_to_peer: {
    task: 'Brainstorm innovative product ideas for AI-powered developer tools',
    peers: ['Alice', 'Bob', 'Carol', 'Dave'],
    rounds: 2,
  },
}

export default function CollaborationPage() {
  const [selectedPattern, setSelectedPattern] = useState(null)
  const [configs, setConfigs] = useState(DEFAULT_CONFIGS)
  const [results, setResults] = useState({})
  const [expandedSteps, setExpandedSteps] = useState({})

  const { data: patternsData } = useQuery({
    queryKey: ['collaboration-patterns'],
    queryFn: async () => {
      const response = await collaborationAPI.getPatterns()
      return response.data
    },
  })

  const consensusMutation = useMutation({
    mutationFn: (data) => collaborationAPI.runConsensus(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, consensus: response.data }))
    },
    onError: (err) => {
      toast.error(`Consensus failed: ${err.message}`)
    },
  })

  const debateMutation = useMutation({
    mutationFn: (data) => collaborationAPI.runDebate(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, debate: response.data }))
    },
    onError: (err) => {
      toast.error(`Debate failed: ${err.message}`)
    },
  })

  const hierarchicalMutation = useMutation({
    mutationFn: (data) => collaborationAPI.runHierarchical(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, hierarchical: response.data }))
    },
    onError: (err) => {
      toast.error(`Hierarchical failed: ${err.message}`)
    },
  })

  const peerToPeerMutation = useMutation({
    mutationFn: (data) => collaborationAPI.runPeerToPeer(data),
    onSuccess: (response) => {
      setResults((prev) => ({ ...prev, peer_to_peer: response.data }))
    },
    onError: (err) => {
      toast.error(`Peer-to-peer failed: ${err.message}`)
    },
  })

  const mutations = {
    consensus: consensusMutation,
    debate: debateMutation,
    hierarchical: hierarchicalMutation,
    peer_to_peer: peerToPeerMutation,
  }

  const handleRun = (patternId) => {
    const config = configs[patternId]
    mutations[patternId].mutate(config)
  }

  const updateConfig = (patternId, field, value) => {
    setConfigs((prev) => ({
      ...prev,
      [patternId]: { ...prev[patternId], [field]: value },
    }))
  }

  const toggleStep = (key) => {
    setExpandedSteps((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const patterns = patternsData?.patterns || []
  const isRunning = (id) => mutations[id]?.isPending

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">
          Agent Collaboration Patterns
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Explore four collaboration patterns from ia_modules: how agents work together to solve problems
        </p>
      </div>

      {/* Pattern Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {patterns.map((pattern) => (
          <button
            key={pattern.id}
            onClick={() => setSelectedPattern(selectedPattern === pattern.id ? null : pattern.id)}
            className={`
              text-left p-5 rounded-xl border-2 transition-all duration-200
              ${selectedPattern === pattern.id
                ? `${PATTERN_BG[pattern.id]} border-opacity-100 shadow-lg scale-[1.02]`
                : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md'
              }
            `}
          >
            <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${PATTERN_COLORS[pattern.id]} text-white mb-3`}>
              {PATTERN_ICONS[pattern.id]}
            </div>
            <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100">{pattern.name}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
              {pattern.description}
            </p>
            <div className="flex flex-wrap gap-1 mt-3">
              {(pattern.use_cases || []).slice(0, 2).map((uc, i) => (
                <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                  {uc}
                </span>
              ))}
            </div>
          </button>
        ))}
      </div>

      {/* Selected Pattern Configuration & Run */}
      {selectedPattern && (
        <div className={`rounded-xl border-2 overflow-hidden ${PATTERN_BG[selectedPattern]}`}>
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">
                Configure & Run: {patterns.find((p) => p.id === selectedPattern)?.name}
              </h2>
              <button
                onClick={() => handleRun(selectedPattern)}
                disabled={isRunning(selectedPattern)}
                className={`
                  flex items-center gap-2 px-5 py-2.5 rounded-xl text-white font-medium
                  bg-gradient-to-r ${PATTERN_COLORS[selectedPattern]}
                  hover:opacity-90 disabled:opacity-50 transition-all shadow-lg
                `}
              >
                {isRunning(selectedPattern) ? (
                  <>
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Run
                  </>
                )}
              </button>
            </div>

            {/* Pattern-specific config forms */}
            {selectedPattern === 'consensus' && (
              <ConsensusConfig config={configs.consensus} onChange={(f, v) => updateConfig('consensus', f, v)} />
            )}
            {selectedPattern === 'debate' && (
              <DebateConfig config={configs.debate} onChange={(f, v) => updateConfig('debate', f, v)} />
            )}
            {selectedPattern === 'hierarchical' && (
              <HierarchicalConfig config={configs.hierarchical} onChange={(f, v) => updateConfig('hierarchical', f, v)} />
            )}
            {selectedPattern === 'peer_to_peer' && (
              <PeerToPeerConfig config={configs.peer_to_peer} onChange={(f, v) => updateConfig('peer_to_peer', f, v)} />
            )}
          </div>
        </div>
      )}

      {/* Results */}
      {selectedPattern && results[selectedPattern] && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="p-6">
            <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100 mb-4">
              Results
            </h2>
            <ResultsDisplay
              result={results[selectedPattern]}
              patternId={selectedPattern}
              expandedSteps={expandedSteps}
              toggleStep={toggleStep}
            />
          </div>
        </div>
      )}
    </div>
  )
}


// ==================== Config Forms ====================

function ConsensusConfig({ config, onChange }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Proposal / Topic</label>
        <textarea
          value={config.topic}
          onChange={(e) => onChange('topic', e.target.value)}
          rows={2}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Agents (comma-separated)</label>
        <input
          type="text"
          value={config.agents.join(', ')}
          onChange={(e) => onChange('agents', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
        />
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Strategy</label>
          <select
            value={config.strategy}
            onChange={(e) => onChange('strategy', e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
          >
            <option value="majority">Majority</option>
            <option value="unanimous">Unanimous</option>
            <option value="supermajority">Supermajority</option>
            <option value="weighted">Weighted</option>
          </select>
        </div>
        <div className="w-32">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Max Iterations</label>
          <input
            type="number"
            min={1}
            max={10}
            value={config.max_iterations}
            onChange={(e) => onChange('max_iterations', parseInt(e.target.value) || 3)}
            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
          />
        </div>
      </div>
    </div>
  )
}

function DebateConfig({ config, onChange }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Debate Topic</label>
        <textarea
          value={config.topic}
          onChange={(e) => onChange('topic', e.target.value)}
          rows={2}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Proponents (comma-separated)</label>
        <input
          type="text"
          value={config.proponents.join(', ')}
          onChange={(e) => onChange('proponents', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Opponents (comma-separated)</label>
        <input
          type="text"
          value={config.opponents.join(', ')}
          onChange={(e) => onChange('opponents', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Moderator</label>
        <input
          type="text"
          value={config.moderator}
          onChange={(e) => onChange('moderator', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500"
        />
      </div>
      <div className="w-32">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Rounds</label>
        <input
          type="number"
          min={1}
          max={5}
          value={config.rounds}
          onChange={(e) => onChange('rounds', parseInt(e.target.value) || 2)}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500"
        />
      </div>
    </div>
  )
}

function HierarchicalConfig({ config, onChange }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Task</label>
        <textarea
          value={config.task}
          onChange={(e) => onChange('task', e.target.value)}
          rows={2}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Leader</label>
        <input
          type="text"
          value={config.leader}
          onChange={(e) => onChange('leader', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Workers (comma-separated)</label>
        <input
          type="text"
          value={config.workers.join(', ')}
          onChange={(e) => onChange('workers', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>
    </div>
  )
}

function PeerToPeerConfig({ config, onChange }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Collaborative Task</label>
        <textarea
          value={config.task}
          onChange={(e) => onChange('task', e.target.value)}
          rows={2}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Peers (comma-separated)</label>
        <input
          type="text"
          value={config.peers.join(', ')}
          onChange={(e) => onChange('peers', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
        />
      </div>
      <div className="w-32">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Rounds</label>
        <input
          type="number"
          min={1}
          max={5}
          value={config.rounds}
          onChange={(e) => onChange('rounds', parseInt(e.target.value) || 2)}
          className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
        />
      </div>
    </div>
  )
}


// ==================== Results Display ====================

function ResultsDisplay({ result, patternId, expandedSteps, toggleStep }) {
  const history = result.history || []

  return (
    <div className="space-y-4">
      {/* Summary Banner */}
      <ResultSummary result={result} patternId={patternId} />

      {/* Step-by-step History */}
      <div>
        <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
          Collaboration Timeline
        </h3>
        <div className="space-y-2">
          {history.map((step, idx) => {
            const key = `${patternId}_${idx}`
            const isExpanded = expandedSteps[key]
            const phaseColor = getPhaseColor(step.phase)

            return (
              <div
                key={key}
                className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
              >
                <button
                  onClick={() => toggleStep(key)}
                  className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition text-left"
                >
                  <div className="flex items-center gap-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${phaseColor}`}>
                      {formatPhase(step.phase)}
                    </span>
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      {step.message || step.phase}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {step.agent && (
                      <span className="text-xs px-2 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                        {step.agent}
                      </span>
                    )}
                    {step.side && (
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        step.side === 'proponent' || step.side === 'proponents'
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                          : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                      }`}>
                        {step.side}
                      </span>
                    )}
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                </button>
                {isExpanded && (
                  <div className="px-4 py-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
                    <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap overflow-x-auto max-h-64 overflow-y-auto">
                      {JSON.stringify(stripMeta(step), null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function ResultSummary({ result, patternId }) {
  if (patternId === 'consensus') {
    const r = result.result || {}
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Consensus"
          value={r.consensus_reached ? 'Reached' : 'Not Reached'}
          icon={r.consensus_reached ? <CheckCircle size={20} className="text-green-500" /> : <XCircle size={20} className="text-red-500" />}
        />
        <SummaryCard
          label="Agreement"
          value={`${((r.agreement_level || 0) * 100).toFixed(0)}%`}
          icon={<Users size={20} className="text-emerald-500" />}
        />
        <SummaryCard
          label="Strategy"
          value={result.strategy || 'N/A'}
          icon={<CheckCircle2 size={20} className="text-teal-500" />}
        />
        <SummaryCard
          label="Iterations"
          value={r.iterations || 0}
          icon={<Clock size={20} className="text-gray-500" />}
        />
      </div>
    )
  }

  if (patternId === 'debate') {
    const r = result.result || {}
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Status"
          value={r.status || 'completed'}
          icon={<CheckCircle size={20} className="text-green-500" />}
        />
        <SummaryCard
          label="Rounds"
          value={r.total_rounds || 0}
          icon={<Shield size={20} className="text-red-500" />}
        />
        <SummaryCard
          label="Arguments"
          value={r.total_arguments || 0}
          icon={<MessageSquare size={20} className="text-orange-500" />}
        />
        <SummaryCard
          label="Key Points"
          value={(r.proponent_key_points?.length || 0) + (r.opponent_key_points?.length || 0)}
          icon={<ArrowRight size={20} className="text-gray-500" />}
        />
      </div>
    )
  }

  if (patternId === 'hierarchical') {
    const r = result.result || {}
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Status"
          value={r.status || 'unknown'}
          icon={r.status === 'success' ? <CheckCircle size={20} className="text-green-500" /> : <XCircle size={20} className="text-yellow-500" />}
        />
        <SummaryCard
          label="Total Workers"
          value={r.total_workers || 0}
          icon={<Users size={20} className="text-blue-500" />}
        />
        <SummaryCard
          label="Successful"
          value={r.successful_workers || 0}
          icon={<CheckCircle size={20} className="text-green-500" />}
        />
        <SummaryCard
          label="Failed"
          value={r.failed_workers || 0}
          icon={<XCircle size={20} className="text-red-500" />}
        />
      </div>
    )
  }

  if (patternId === 'peer_to_peer') {
    const r = result.result || {}
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Status"
          value={r.status || 'unknown'}
          icon={<CheckCircle size={20} className="text-green-500" />}
        />
        <SummaryCard
          label="Peers"
          value={r.total_peers || 0}
          icon={<Users size={20} className="text-purple-500" />}
        />
        <SummaryCard
          label="Contributions"
          value={r.total_contributions || 0}
          icon={<MessageSquare size={20} className="text-pink-500" />}
        />
        <SummaryCard
          label="Rounds"
          value={r.rounds_completed || 0}
          icon={<Clock size={20} className="text-gray-500" />}
        />
      </div>
    )
  }

  return null
}

function SummaryCard({ label, value, icon }) {
  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
      {icon}
      <div>
        <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
        <p className="text-sm font-bold text-gray-800 dark:text-gray-200 capitalize">{String(value)}</p>
      </div>
    </div>
  )
}


// ==================== Helpers ====================

function getPhaseColor(phase) {
  if (phase.includes('setup')) return 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
  if (phase.includes('initialization')) return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
  if (phase.includes('voting') || phase.includes('vote')) return 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
  if (phase.includes('proposal')) return 'bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-400'
  if (phase.includes('opening')) return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
  if (phase.includes('closing')) return 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400'
  if (phase.includes('argument') || phase.includes('round')) return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400'
  if (phase.includes('result')) return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
  if (phase.includes('error')) return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
  if (phase.includes('worker') || phase.includes('task')) return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
  if (phase.includes('contribution')) return 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400'
  return 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
}

function formatPhase(phase) {
  return phase
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function stripMeta(step) {
  const { phase, message, timestamp, ...rest } = step
  return rest
}

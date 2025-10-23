import React from 'react'
import { Link } from 'react-router-dom'
import { Play, BarChart3, CheckCircle, Clock, TrendingUp, Shield } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-500 rounded-lg p-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Welcome to IA Modules Showcase</h1>
        <p className="text-xl mb-6">
          Production-ready AI agent framework with enterprise-grade reliability and observability
        </p>
        <div className="flex space-x-4">
          <Link
            to="/pipelines"
            className="bg-white text-primary-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition"
          >
            Try Example Pipelines
          </Link>
          <Link
            to="/metrics"
            className="border-2 border-white px-6 py-3 rounded-lg font-semibold hover:bg-white hover:text-primary-600 transition"
          >
            View Metrics Dashboard
          </Link>
        </div>
      </div>

      {/* Features Grid */}
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            icon={<Play className="text-primary-500" size={32} />}
            title="Graph-Based Pipelines"
            description="Define complex workflows as directed graphs with conditional routing and parallel execution"
          />
          <FeatureCard
            icon={<BarChart3 className="text-primary-500" size={32} />}
            title="12 Reliability Metrics"
            description="Track SVR, CR, PC, HIR, MA, MTTE, RSR, EQS, TCL, WCT, TPW, CPSW with real-time monitoring"
          />
          <FeatureCard
            icon={<Shield className="text-primary-500" size={32} />}
            title="EARF Three Pillars"
            description="Total Observability, Absolute Reproducibility, Formal Safety & Verification"
          />
          <FeatureCard
            icon={<CheckCircle className="text-primary-500" size={32} />}
            title="Human-in-the-Loop"
            description="Pause-and-resume workflows with human approval gates and collaborative decision making"
          />
          <FeatureCard
            icon={<Clock className="text-primary-500" size={32} />}
            title="Checkpointing & Resume"
            description="Automatic state snapshots enable resuming failed pipelines from last successful step"
          />
          <FeatureCard
            icon={<TrendingUp className="text-primary-500" size={32} />}
            title="Memory & Conversation"
            description="Context-aware processing with conversation history and session management"
          />
          <FeatureCard
            icon={<Play className="text-primary-500" size={32} />}
            title="Advanced Routing"
            description="Conditional branching, parallel execution, and loop detection for complex workflows"
          />
        </div>
      </div>

      {/* Stats */}
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Framework Capabilities</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <StatCard label="Example Pipelines" value="9" color="blue" />
          <StatCard label="Reliability Metrics" value="12" color="purple" />
          <StatCard label="EARF Pillars" value="3" color="green" />
          <StatCard label="Python Support" value="3.9-3.13" color="blue" />
        </div>
      </div>

      {/* Quick Start */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Quick Start</h2>
        <div className="space-y-4">
          <QuickStartStep
            number="1"
            title="Explore Example Pipelines"
            description="Navigate to Pipelines to see pre-built examples demonstrating framework capabilities"
          />
          <QuickStartStep
            number="2"
            title="Run a Pipeline"
            description="Click 'Execute' on any pipeline to see real-time execution with live metrics tracking"
          />
          <QuickStartStep
            number="3"
            title="Monitor Reliability"
            description="View the Metrics Dashboard to see comprehensive reliability metrics and SLO compliance"
          />
          <QuickStartStep
            number="4"
            title="Review Execution History"
            description="Check Executions tab to see all pipeline runs with detailed logs and results"
          />
        </div>
      </div>

      {/* Documentation Links */}
      <div className="bg-gray-100 rounded-lg p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Documentation</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DocLink
            title="Getting Started"
            description="Quick start guide and installation"
            href="/docs/GETTING_STARTED.md"
          />
          <DocLink
            title="API Reference"
            description="Complete API documentation"
            href="/docs/API_REFERENCE.md"
          />
          <DocLink
            title="Reliability Guide"
            description="EARF compliance and monitoring"
            href="/docs/RELIABILITY_USAGE_GUIDE.md"
          />
          <DocLink
            title="Features Overview"
            description="Complete feature matrix"
            href="/docs/FEATURES.md"
          />
        </div>
      </div>
    </div>
  )
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="bg-white rounded-lg shadow p-6 hover:shadow-lg transition">
      <div className="mb-4">{icon}</div>
      <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </div>
  )
}

function StatCard({ label, value, color }) {
  const colors = {
    green: 'bg-green-100 text-green-800',
    blue: 'bg-blue-100 text-blue-800',
    purple: 'bg-purple-100 text-purple-800',
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="text-sm text-gray-600 mb-2">{label}</div>
      <div className={`text-3xl font-bold ${colors[color]}`}>{value}</div>
    </div>
  )
}

function QuickStartStep({ number, title, description }) {
  return (
    <div className="flex items-start space-x-4">
      <div className="bg-primary-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
        {number}
      </div>
      <div>
        <h3 className="font-semibold text-gray-800">{title}</h3>
        <p className="text-gray-600 text-sm">{description}</p>
      </div>
    </div>
  )
}

function DocLink({ title, description, href }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="block bg-white rounded-lg p-4 hover:shadow-md transition"
    >
      <h3 className="font-semibold text-gray-800 mb-1">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </a>
  )
}

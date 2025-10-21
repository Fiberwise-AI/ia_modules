-- Reliability Metrics Schema
-- Migration V007: Add tables for reliability metrics, SLO tracking, and anomaly detection

-- Reliability step metrics
CREATE TABLE IF NOT EXISTS reliability_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    required_compensation BOOLEAN DEFAULT FALSE,
    required_human BOOLEAN DEFAULT FALSE,
    mode TEXT,
    declared_mode TEXT,
    mode_violation BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes for common queries
    INDEX idx_reliability_steps_agent (agent_name),
    INDEX idx_reliability_steps_timestamp (timestamp),
    INDEX idx_reliability_steps_success (success)
);

-- Reliability workflow metrics
CREATE TABLE IF NOT EXISTS reliability_workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL UNIQUE,
    steps INTEGER NOT NULL,
    retries INTEGER DEFAULT 0,
    success BOOLEAN NOT NULL,
    required_human BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_reliability_workflows_timestamp (timestamp),
    INDEX idx_reliability_workflows_success (success)
);

-- SLO measurements (MTTE and RSR)
CREATE TABLE IF NOT EXISTS reliability_slo_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_type TEXT NOT NULL CHECK (measurement_type IN ('mtte', 'rsr')),
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT,

    -- MTTE specific
    duration_ms INTEGER,

    -- RSR specific
    replay_mode TEXT,

    -- Common
    success BOOLEAN NOT NULL,
    error TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_slo_measurements_type (measurement_type),
    INDEX idx_slo_measurements_timestamp (timestamp),
    INDEX idx_slo_measurements_thread (thread_id)
);

-- Anomaly detection log
CREATE TABLE IF NOT EXISTS reliability_anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    expected_value REAL,
    actual_value REAL,
    deviation REAL,
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    details TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_anomalies_type (anomaly_type),
    INDEX idx_anomalies_metric (metric_name),
    INDEX idx_anomalies_timestamp (timestamp),
    INDEX idx_anomalies_severity (severity)
);

-- Cost tracking
CREATE TABLE IF NOT EXISTS reliability_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('agent', 'workflow', 'tool')),
    entity_id TEXT NOT NULL,
    cost_type TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT DEFAULT 'USD',
    metadata TEXT, -- JSON
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_costs_entity (entity_type, entity_id),
    INDEX idx_costs_timestamp (timestamp),
    INDEX idx_costs_type (cost_type)
);

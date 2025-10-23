-- Complete Database Schema - Consolidated
-- All tables for ia_modules pipeline system

-- ===========================
-- CORE PIPELINE TABLES
-- ===========================

CREATE TABLE IF NOT EXISTS pipelines (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT DEFAULT '1.0',
    pipeline_json TEXT,
    config_json TEXT,
    file_path TEXT,
    content_hash TEXT,
    is_system INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_templates (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT DEFAULT 'example',
    pipeline_json TEXT,
    config_json TEXT,
    is_system INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_executions (
    execution_id TEXT PRIMARY KEY,
    pipeline_id TEXT,
    pipeline_slug TEXT,
    pipeline_name TEXT,
    user_id INTEGER,
    status TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    total_steps INTEGER DEFAULT 0,
    completed_steps INTEGER DEFAULT 0,
    failed_steps INTEGER DEFAULT 0,
    input_data TEXT,
    output_data TEXT,
    current_step TEXT,
    error_message TEXT,
    execution_time_ms INTEGER,
    duration_seconds REAL,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS step_executions (
    step_execution_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_type TEXT,
    status TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    input_data TEXT,
    output_data TEXT,
    result_json TEXT,
    error_message TEXT,
    execution_time_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    metadata_json TEXT,
    step_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
);

CREATE TABLE IF NOT EXISTS pipeline_logs (
    id SERIAL PRIMARY KEY,
    execution_id TEXT,
    job_id TEXT,
    step_id TEXT,
    step_name TEXT,
    event_type TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data TEXT,
    duration_ms INTEGER
);

CREATE TABLE IF NOT EXISTS execution_logs (
    id SERIAL PRIMARY KEY,
    execution_id TEXT NOT NULL,
    step_id TEXT,
    log_level TEXT DEFAULT 'INFO',
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
);

-- ===========================
-- CHECKPOINT SYSTEM
-- ===========================

CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL,
    pipeline_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) NOT NULL,
    step_index INTEGER NOT NULL,
    step_name VARCHAR(255),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    state JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'completed',
    parent_checkpoint_id UUID,
    CONSTRAINT fk_parent_checkpoint FOREIGN KEY (parent_checkpoint_id) REFERENCES pipeline_checkpoints(checkpoint_id) ON DELETE SET NULL
);

-- ===========================
-- CONVERSATION MEMORY
-- ===========================

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    function_call JSONB,
    function_name VARCHAR(255)
);

-- ===========================
-- RELIABILITY METRICS
-- ===========================

CREATE TABLE IF NOT EXISTS reliability_steps (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    required_compensation BOOLEAN DEFAULT FALSE,
    required_human BOOLEAN DEFAULT FALSE,
    mode VARCHAR(50),
    declared_mode VARCHAR(50),
    mode_violation BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reliability_workflows (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL UNIQUE,
    steps INTEGER NOT NULL,
    retries INTEGER DEFAULT 0,
    success BOOLEAN NOT NULL,
    required_human BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reliability_slo_measurements (
    id SERIAL PRIMARY KEY,
    measurement_type VARCHAR(10) NOT NULL CHECK (measurement_type IN ('mtte', 'rsr')),
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255),
    duration_ms INTEGER,
    replay_mode VARCHAR(50),
    success BOOLEAN NOT NULL,
    error TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reliability_anomalies (
    id SERIAL PRIMARY KEY,
    anomaly_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    expected_value REAL,
    actual_value REAL,
    deviation REAL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    details TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reliability_costs (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('agent', 'workflow', 'tool')),
    entity_id VARCHAR(255) NOT NULL,
    cost_type VARCHAR(100) NOT NULL,
    amount REAL NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    metadata JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ===========================
-- INDEXES
-- ===========================

CREATE INDEX IF NOT EXISTS idx_pipelines_slug ON pipelines(slug);
CREATE INDEX IF NOT EXISTS idx_pipelines_active ON pipelines(is_active);
CREATE INDEX IF NOT EXISTS idx_pipeline_templates_slug ON pipeline_templates(slug);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_execution ON pipeline_logs(execution_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_pipeline_id ON pipeline_executions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status ON pipeline_executions(status);
CREATE INDEX IF NOT EXISTS idx_step_executions_execution_id ON step_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_step_executions_status ON step_executions(status);
CREATE INDEX IF NOT EXISTS idx_execution_logs_execution_id ON execution_logs(execution_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_id ON pipeline_checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_pipeline_id ON pipeline_checkpoints(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp ON pipeline_checkpoints(timestamp);

CREATE INDEX IF NOT EXISTS idx_msg_thread_id ON conversation_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON conversation_messages(timestamp);

CREATE INDEX IF NOT EXISTS idx_reliability_steps_agent ON reliability_steps(agent_name);
CREATE INDEX IF NOT EXISTS idx_reliability_steps_timestamp ON reliability_steps(timestamp);
CREATE INDEX IF NOT EXISTS idx_reliability_workflows_timestamp ON reliability_workflows(timestamp);
CREATE INDEX IF NOT EXISTS idx_slo_measurements_type ON reliability_slo_measurements(measurement_type);
CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON reliability_anomalies(timestamp);
CREATE INDEX IF NOT EXISTS idx_costs_entity ON reliability_costs(entity_type, entity_id);

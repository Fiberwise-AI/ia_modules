BEGIN TRANSACTION;

-- Consolidated Pipeline & Execution Schema (V002)
-- This migration contains the canonical tables used by apps and the web interface.
-- It merges the earlier V002 and V003 variants into one coherent schema.

-- Core pipelines table (canonical pipeline definitions)
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

-- Templates / examples table (shared for CLI/web)
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

-- Per-step logs and events (legacy; kept for compatibility)
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT,
    job_id TEXT,
    step_id TEXT,
    step_name TEXT,
    event_type TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data TEXT,
    duration_ms INTEGER
);

-- Enhanced execution tracker table (primary execution tracking system)
-- execution_id is the canonical identifier used in code paths; keep as TEXT primary key
CREATE TABLE IF NOT EXISTS pipeline_executions (
    execution_id TEXT PRIMARY KEY,
    id INTEGER UNIQUE,
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

-- Detailed step executions (enhanced)
CREATE TABLE IF NOT EXISTS step_executions_enhanced (
    step_execution_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_type TEXT,
    status TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    input_data TEXT,
    output_data TEXT,
    result_json TEXT,
    error_message TEXT,
    execution_time_ms INTEGER,
    duration_seconds REAL,
    retry_count INTEGER DEFAULT 0,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
);

-- Simpler legacy step_executions table (kept for apps that expect it)
CREATE TABLE IF NOT EXISTS step_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    result_json TEXT,
    error_message TEXT,
    step_order INTEGER,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
);

-- Detailed execution logs for the web interface
CREATE TABLE IF NOT EXISTS execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    step_id TEXT,
    log_level TEXT DEFAULT 'INFO',
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
);

-- Indexes for performance and common lookups
CREATE INDEX IF NOT EXISTS idx_pipelines_slug ON pipelines(slug);
CREATE INDEX IF NOT EXISTS idx_pipelines_active ON pipelines(is_active);
CREATE INDEX IF NOT EXISTS idx_pipeline_templates_slug ON pipeline_templates(slug);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_execution ON pipeline_logs(execution_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_pipeline_id ON pipeline_executions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_pipeline_slug ON pipeline_executions(pipeline_slug);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status ON pipeline_executions(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_started_at ON pipeline_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_step_executions_execution_id ON step_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_step_executions_enhanced_execution_id ON step_executions_enhanced(execution_id);
CREATE INDEX IF NOT EXISTS idx_step_executions_enhanced_status ON step_executions_enhanced(status);
CREATE INDEX IF NOT EXISTS idx_execution_logs_execution_id ON execution_logs(execution_id);

COMMIT;
-- Pipeline Management System - V002
-- Description: Add comprehensive pipeline management tables
-- Dependencies: V001__complete_schema.sql

-- Pipelines table for storing pipeline definitions
CREATE TABLE IF NOT EXISTS pipelines (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT DEFAULT '1.0',
    pipeline_json TEXT NOT NULL,
    file_path TEXT,  -- Relative path to JSON file
    content_hash TEXT,  -- Hash of pipeline content for change detection
    is_active BOOLEAN DEFAULT 1,
    is_system BOOLEAN DEFAULT 0,  -- System pipelines vs user pipelines
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline executions table (separate from user-specific executions)
CREATE TABLE IF NOT EXISTS pipeline_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT UNIQUE NOT NULL,
    pipeline_id TEXT NOT NULL,
    pipeline_slug TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    user_id INTEGER,  -- NULL for system executions
    status TEXT NOT NULL DEFAULT 'pending',
    input_data TEXT,  -- JSON string
    output_data TEXT, -- JSON string
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipelines (id),
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
);

-- Step executions table for detailed tracking
CREATE TABLE IF NOT EXISTS step_executions_enhanced (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    step_execution_id TEXT UNIQUE NOT NULL,  -- UUID for each step execution
    step_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_type TEXT NOT NULL DEFAULT 'task',  -- Type of step (task, conditional, etc.)
    status TEXT NOT NULL,
    input_data TEXT,  -- JSON string
    output_data TEXT, -- JSON string for step results
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_ms INTEGER,
    step_order INTEGER,
    retry_count INTEGER DEFAULT 0,
    metadata_json TEXT, -- Additional metadata as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id) ON DELETE CASCADE
);

-- Pipeline templates/examples table
CREATE TABLE IF NOT EXISTS pipeline_templates (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT DEFAULT 'user',
    pipeline_json TEXT NOT NULL,
    is_system BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_pipelines_slug ON pipelines(slug);
CREATE INDEX IF NOT EXISTS idx_pipelines_active ON pipelines(is_active);
CREATE INDEX IF NOT EXISTS idx_pipelines_system ON pipelines(is_system);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_pipeline_id ON pipeline_executions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_user_id ON pipeline_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status ON pipeline_executions(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_started ON pipeline_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_step_executions_execution_id ON step_executions_enhanced(execution_id);
CREATE INDEX IF NOT EXISTS idx_step_executions_step_id ON step_executions_enhanced(step_id);
CREATE INDEX IF NOT EXISTS idx_step_executions_status ON step_executions_enhanced(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_templates_category ON pipeline_templates(category);
CREATE INDEX IF NOT EXISTS idx_pipeline_templates_system ON pipeline_templates(is_system);

-- Note: Skipping data migration for now - starting fresh with new pipeline system

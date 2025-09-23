-- Migration: create pipelines, pipeline_jobs and pipeline_logs tables

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS pipelines (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    name TEXT,
    description TEXT,
    version TEXT,
    pipeline_json TEXT,
    file_path TEXT,
    content_hash TEXT,
    is_system INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pipelines_slug ON pipelines(slug);

CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id TEXT PRIMARY KEY,
    pipeline_slug TEXT,
    status TEXT,
    started_at DATETIME,
    finished_at DATETIME,
    current_step TEXT,
    step_statuses TEXT,
    input_data TEXT,
    result TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_slug ON pipeline_jobs(pipeline_slug);

CREATE TABLE IF NOT EXISTS pipeline_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    step_name TEXT,
    event_type TEXT,
    timestamp DATETIME,
    data TEXT,
    duration REAL
);

CREATE INDEX IF NOT EXISTS idx_pipeline_logs_job ON pipeline_logs(job_id);

COMMIT;

-- Complete Database Schema - Single Source of Truth
-- Version: V001
-- Description: All tables needed for the pipeline system

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    first_name TEXT,
    last_name TEXT,
    name TEXT,
    role TEXT DEFAULT 'user',
    active BOOLEAN DEFAULT 1,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table for authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Pipeline configurations table
CREATE TABLE IF NOT EXISTS pipeline_configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    pipeline_json TEXT NOT NULL,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Pipeline step logging
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('start', 'complete', 'error')),
    timestamp TEXT NOT NULL,
    data TEXT,
    duration REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create essential indexes
CREATE INDEX IF NOT EXISTS idx_users_uuid ON users(uuid);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_job_id ON pipeline_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_step_name ON pipeline_logs(step_name);

-- Wiki pages table for knowledge base
CREATE TABLE IF NOT EXISTS wiki_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT UNIQUE NOT NULL,
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    content_json TEXT NOT NULL, -- Full wiki page structure as JSON
    word_count INTEGER DEFAULT 0,
    sections_count INTEGER DEFAULT 0,
    quality_score INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    is_current_version BOOLEAN DEFAULT 1,
    parent_version_id INTEGER, -- Reference to previous version
    source_data_json TEXT, -- Metadata about sources used
    change_summary TEXT, -- Summary of what changed in this version
    created_by TEXT DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_version_id) REFERENCES wiki_pages(id)
);

-- Wiki page revision history table
CREATE TABLE IF NOT EXISTS wiki_page_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    content_json TEXT NOT NULL,
    change_summary TEXT,
    created_by TEXT DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES wiki_pages(page_id)
);

-- Wiki pages indexes
CREATE INDEX IF NOT EXISTS idx_wiki_pages_page_id ON wiki_pages(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_topic ON wiki_pages(topic);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_title ON wiki_pages(title);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_created_at ON wiki_pages(created_at);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_version ON wiki_pages(version);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_current ON wiki_pages(is_current_version);
CREATE INDEX IF NOT EXISTS idx_wiki_revisions_page_id ON wiki_page_revisions(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_revisions_version ON wiki_page_revisions(version);

-- Default system user (optional user_id for pipeline executions)
INSERT OR IGNORE INTO users (uuid, email, name, role, active) VALUES
    ('system-default-user', 'system@pipeline.local', 'System Default User', 'system', 1);
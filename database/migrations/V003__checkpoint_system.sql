-- Migration V003: Checkpoint System
-- Adds checkpoint storage for pipeline pause/resume functionality

-- Create pipeline_checkpoints table
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    pipeline_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    step_name TEXT,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL,  -- JSON serialized state
    metadata TEXT DEFAULT '{}',  -- JSON serialized metadata
    status TEXT DEFAULT 'completed',
    parent_checkpoint_id TEXT,

    FOREIGN KEY (parent_checkpoint_id)
        REFERENCES pipeline_checkpoints(checkpoint_id)
        ON DELETE SET NULL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_id
    ON pipeline_checkpoints(thread_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_pipeline_id
    ON pipeline_checkpoints(pipeline_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp
    ON pipeline_checkpoints(timestamp);

CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_timestamp
    ON pipeline_checkpoints(thread_id, timestamp);

-- Note: For PostgreSQL, use the schema in checkpoint/sql.py which has JSONB and UUID types

-- Structured Content Database Schema Migration
-- This file contains the SQL statements to create tables for structured page processing

-- Pages table for storing processed content
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    markdown_content TEXT,
    markdown_file_path TEXT,
    last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sections for hierarchical content structure
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    header TEXT NOT NULL,
    level INTEGER NOT NULL,
    section_order INTEGER NOT NULL,
    section_id TEXT,
    header_id TEXT,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES pages (id) ON DELETE CASCADE
);

-- Paragraphs within sections
CREATE TABLE IF NOT EXISTS paragraphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    processed_text TEXT,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (section_id) REFERENCES sections (id) ON DELETE CASCADE
);

-- Individual sentences for granular analysis
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paragraph_id INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    sentence_order INTEGER NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paragraph_id) REFERENCES paragraphs (id) ON DELETE CASCADE
);

-- Links within sentences for relationship mapping
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    url TEXT NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sentence_id) REFERENCES sentences (id) ON DELETE CASCADE
);

-- Section notes for metadata
CREATE TABLE IF NOT EXISTS section_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id INTEGER NOT NULL,
    note_text TEXT NOT NULL,
    note_urls TEXT,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (section_id) REFERENCES sections (id) ON DELETE CASCADE
);

-- Processing requests for audit trail
CREATE TABLE IF NOT EXISTS page_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title);
CREATE INDEX IF NOT EXISTS idx_sections_page_id ON sections(page_id);
CREATE INDEX IF NOT EXISTS idx_sections_header ON sections(header);
CREATE INDEX IF NOT EXISTS idx_paragraphs_section_id ON paragraphs(section_id);
CREATE INDEX IF NOT EXISTS idx_sentences_paragraph_id ON sentences(paragraph_id);
CREATE INDEX IF NOT EXISTS idx_links_sentence_id ON links(sentence_id);
CREATE INDEX IF NOT EXISTS idx_links_url ON links(url);
CREATE INDEX IF NOT EXISTS idx_section_notes_section_id ON section_notes(section_id);
CREATE INDEX IF NOT EXISTS idx_page_requests_user_id ON page_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_page_requests_agent_id ON page_requests(agent_id);

-- Full-text search indexes for content searching
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    title,
    markdown_content,
    content='pages',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts USING fts5(
    original_text,
    processed_text,
    content='paragraphs',
    content_rowid='id'
);

-- Triggers to maintain FTS indexes
CREATE TRIGGER IF NOT EXISTS pages_fts_insert AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, title, markdown_content)
    VALUES (new.id, new.title, new.markdown_content);
END;

CREATE TRIGGER IF NOT EXISTS pages_fts_update AFTER UPDATE ON pages BEGIN
    UPDATE pages_fts SET title = new.title, markdown_content = new.markdown_content
    WHERE rowid = new.id;
END;

CREATE TRIGGER IF NOT EXISTS pages_fts_delete AFTER DELETE ON pages BEGIN
    DELETE FROM pages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS paragraphs_fts_insert AFTER INSERT ON paragraphs BEGIN
    INSERT INTO paragraphs_fts(rowid, original_text, processed_text)
    VALUES (new.id, new.original_text, new.processed_text);
END;

CREATE TRIGGER IF NOT EXISTS paragraphs_fts_update AFTER UPDATE ON paragraphs BEGIN
    UPDATE paragraphs_fts SET original_text = new.original_text, processed_text = new.processed_text
    WHERE rowid = new.id;
END;

CREATE TRIGGER IF NOT EXISTS paragraphs_fts_delete AFTER DELETE ON paragraphs BEGIN
    DELETE FROM paragraphs_fts WHERE rowid = old.id;
END;
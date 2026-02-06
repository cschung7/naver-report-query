-- Economic Analysis Reports Schema

CREATE TABLE IF NOT EXISTS econ_reports (
    report_id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    issuer VARCHAR(100),
    issue_date DATE,
    author VARCHAR(100),
    email VARCHAR(100),
    summary TEXT,

    -- Economic metadata
    category VARCHAR(50),
    region VARCHAR(50),

    -- Economic indicators
    indicator_type VARCHAR(50),
    forecast_period VARCHAR(50),

    -- Key metrics
    key_metrics JSONB,
    key_themes JSONB,

    -- File references
    pdf_link TEXT,
    local_file TEXT,
    marker_file TEXT,
    extraction_file TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_econ_issue_date ON econ_reports(issue_date DESC);
CREATE INDEX IF NOT EXISTS idx_econ_issuer ON econ_reports(issuer);
CREATE INDEX IF NOT EXISTS idx_econ_category ON econ_reports(category);
CREATE INDEX IF NOT EXISTS idx_econ_region ON econ_reports(region);

-- Full text search
CREATE INDEX IF NOT EXISTS idx_econ_title_gin ON econ_reports USING gin(to_tsvector('simple', title));
CREATE INDEX IF NOT EXISTS idx_econ_summary_gin ON econ_reports USING gin(to_tsvector('simple', COALESCE(summary, '')));

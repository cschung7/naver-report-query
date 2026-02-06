-- Investment Strategy Reports Schema
-- Separate from FirmAnalysis reports

-- Main reports table
CREATE TABLE IF NOT EXISTS invest_strategy_reports (
    report_id VARCHAR(500) PRIMARY KEY,
    title TEXT NOT NULL,
    issuer VARCHAR(100) NOT NULL,
    issue_date DATE NOT NULL,
    author TEXT,
    email TEXT,
    summary TEXT,

    -- Extraction metadata
    strategy_type TEXT,
    time_horizon TEXT,
    geography TEXT,
    extraction_confidence FLOAT,

    -- Market view
    market_outlook TEXT,
    conviction_level TEXT,
    market_stage TEXT,
    market_regime TEXT,
    key_thesis JSONB,

    -- Market positioning
    equity_allocation TEXT,
    bond_allocation TEXT,
    cash_allocation TEXT,
    risk_posture TEXT,

    -- Macro backdrop
    gdp_trend TEXT,
    growth_drivers JSONB,
    growth_risks JSONB,
    monetary_policy_stance TEXT,
    rate_direction TEXT,

    -- Actionable summary
    do_list JSONB,
    avoid_list JSONB,
    watch_list JSONB,

    -- File references
    pdf_link TEXT,
    local_file TEXT,
    marker_file TEXT,
    extraction_file TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_invest_issue_date ON invest_strategy_reports(issue_date DESC);
CREATE INDEX IF NOT EXISTS idx_invest_issuer ON invest_strategy_reports(issuer);
CREATE INDEX IF NOT EXISTS idx_invest_strategy_type ON invest_strategy_reports(strategy_type);
CREATE INDEX IF NOT EXISTS idx_invest_market_outlook ON invest_strategy_reports(market_outlook);
CREATE INDEX IF NOT EXISTS idx_invest_market_regime ON invest_strategy_reports(market_regime);
CREATE INDEX IF NOT EXISTS idx_invest_geography ON invest_strategy_reports(geography);
CREATE INDEX IF NOT EXISTS idx_invest_date_issuer ON invest_strategy_reports(issue_date DESC, issuer);

-- Full text search
CREATE INDEX IF NOT EXISTS idx_invest_title_fts ON invest_strategy_reports USING gin(to_tsvector('simple', title));
CREATE INDEX IF NOT EXISTS idx_invest_summary_fts ON invest_strategy_reports USING gin(to_tsvector('simple', coalesce(summary, '')));

-- Asset allocation table (for detailed allocations)
CREATE TABLE IF NOT EXISTS invest_allocations (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(500) REFERENCES invest_strategy_reports(report_id) ON DELETE CASCADE,
    asset_category TEXT NOT NULL,
    allocation_weight TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alloc_report ON invest_allocations(report_id);
CREATE INDEX IF NOT EXISTS idx_alloc_category ON invest_allocations(asset_category);

-- Sector recommendations table
CREATE TABLE IF NOT EXISTS invest_sector_recs (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(500) REFERENCES invest_strategy_reports(report_id) ON DELETE CASCADE,
    sector TEXT NOT NULL,
    recommendation TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sector_report ON invest_sector_recs(report_id);
CREATE INDEX IF NOT EXISTS idx_sector_name ON invest_sector_recs(sector);

-- Thematic plays table
CREATE TABLE IF NOT EXISTS invest_themes (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(500) REFERENCES invest_strategy_reports(report_id) ON DELETE CASCADE,
    theme_name TEXT NOT NULL,
    conviction TEXT,
    time_horizon TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_theme_report ON invest_themes(report_id);
CREATE INDEX IF NOT EXISTS idx_theme_name ON invest_themes(theme_name);

-- View for quick report lookup
CREATE OR REPLACE VIEW v_invest_reports_recent AS
SELECT
    report_id,
    title,
    issuer,
    issue_date,
    strategy_type,
    market_outlook,
    market_regime,
    geography,
    extraction_confidence
FROM invest_strategy_reports
WHERE issue_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY issue_date DESC;

-- View for issuer coverage stats
CREATE OR REPLACE VIEW v_invest_issuer_stats AS
SELECT
    issuer,
    COUNT(*) as report_count,
    MIN(issue_date) as first_report,
    MAX(issue_date) as last_report,
    COUNT(DISTINCT strategy_type) as strategy_types
FROM invest_strategy_reports
GROUP BY issuer
ORDER BY report_count DESC;

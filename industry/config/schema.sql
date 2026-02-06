-- Industry Analysis Reports Schema
-- For Korean securities broker industry research reports

-- Main reports table
CREATE TABLE IF NOT EXISTS industry_reports (
    report_id VARCHAR(500) PRIMARY KEY,
    title TEXT NOT NULL,
    issuer VARCHAR(100) NOT NULL,
    issue_date DATE NOT NULL,
    author TEXT,
    email TEXT,
    summary TEXT,

    -- Industry metadata
    industry TEXT,
    sector TEXT,
    geography TEXT,
    extraction_confidence FLOAT,

    -- Cycle analysis
    cycle_stage TEXT,  -- UPCYCLE, DOWNCYCLE, PEAK, TROUGH, RECOVERY
    cycle_drivers JSONB,
    cycle_duration TEXT,

    -- Demand/Supply analysis
    demand_trend TEXT,  -- GROWING, STABLE, CONTRACTING
    demand_drivers JSONB,
    supply_dynamics TEXT,
    competitive_landscape TEXT,

    -- Investment implications
    investment_timing TEXT,  -- BUY, HOLD, AVOID
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

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_industry_issue_date ON industry_reports(issue_date DESC);
CREATE INDEX IF NOT EXISTS idx_industry_issuer ON industry_reports(issuer);
CREATE INDEX IF NOT EXISTS idx_industry_name ON industry_reports(industry);
CREATE INDEX IF NOT EXISTS idx_industry_sector ON industry_reports(sector);
CREATE INDEX IF NOT EXISTS idx_industry_cycle_stage ON industry_reports(cycle_stage);
CREATE INDEX IF NOT EXISTS idx_industry_demand_trend ON industry_reports(demand_trend);
CREATE INDEX IF NOT EXISTS idx_industry_investment_timing ON industry_reports(investment_timing);
CREATE INDEX IF NOT EXISTS idx_industry_geography ON industry_reports(geography);
CREATE INDEX IF NOT EXISTS idx_industry_date_issuer ON industry_reports(issue_date DESC, issuer);

-- Full text search
CREATE INDEX IF NOT EXISTS idx_industry_title_fts ON industry_reports USING gin(to_tsvector('simple', title));
CREATE INDEX IF NOT EXISTS idx_industry_summary_fts ON industry_reports USING gin(to_tsvector('simple', coalesce(summary, '')));

-- Industry themes table
CREATE TABLE IF NOT EXISTS industry_themes (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(500) REFERENCES industry_reports(report_id) ON DELETE CASCADE,
    theme_name TEXT NOT NULL,
    relevance TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_industry_theme_report ON industry_themes(report_id);
CREATE INDEX IF NOT EXISTS idx_industry_theme_name ON industry_themes(theme_name);

-- End market analysis table
CREATE TABLE IF NOT EXISTS industry_end_markets (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(500) REFERENCES industry_reports(report_id) ON DELETE CASCADE,
    market_name TEXT NOT NULL,
    trend TEXT,
    outlook TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_end_market_report ON industry_end_markets(report_id);
CREATE INDEX IF NOT EXISTS idx_end_market_name ON industry_end_markets(market_name);

-- Competitive players table
CREATE TABLE IF NOT EXISTS industry_players (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(500) REFERENCES industry_reports(report_id) ON DELETE CASCADE,
    company_name TEXT NOT NULL,
    position TEXT,
    recommendation TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_player_report ON industry_players(report_id);
CREATE INDEX IF NOT EXISTS idx_player_name ON industry_players(company_name);

-- View for quick report lookup
CREATE OR REPLACE VIEW v_industry_reports_recent AS
SELECT
    report_id,
    title,
    issuer,
    issue_date,
    industry,
    sector,
    cycle_stage,
    demand_trend,
    investment_timing,
    extraction_confidence
FROM industry_reports
WHERE issue_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY issue_date DESC;

-- View for industry coverage stats
CREATE OR REPLACE VIEW v_industry_stats AS
SELECT
    industry,
    COUNT(*) as report_count,
    MIN(issue_date) as first_report,
    MAX(issue_date) as last_report,
    COUNT(DISTINCT issuer) as issuer_count
FROM industry_reports
WHERE industry IS NOT NULL
GROUP BY industry
ORDER BY report_count DESC;

-- View for issuer stats
CREATE OR REPLACE VIEW v_industry_issuer_stats AS
SELECT
    issuer,
    COUNT(*) as report_count,
    MIN(issue_date) as first_report,
    MAX(issue_date) as last_report,
    COUNT(DISTINCT industry) as industries_covered
FROM industry_reports
GROUP BY issuer
ORDER BY report_count DESC;

-- View for cycle distribution
CREATE OR REPLACE VIEW v_industry_cycle_distribution AS
SELECT
    cycle_stage,
    COUNT(*) as report_count,
    COUNT(DISTINCT industry) as industries
FROM industry_reports
WHERE cycle_stage IS NOT NULL
GROUP BY cycle_stage
ORDER BY report_count DESC;

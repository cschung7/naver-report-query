-- NaverReport FirmAnalysis Schema
-- PostgreSQL DDL for firm analysis reports

-- ===========================================
-- Drop existing tables (for clean setup)
-- ===========================================
DROP TABLE IF EXISTS firm_extraction CASCADE;
DROP TABLE IF EXISTS firm_reports CASCADE;

-- ===========================================
-- Core Reports Table
-- ===========================================
-- Source: df_firm.csv + with_author.csv
CREATE TABLE firm_reports (
    -- Primary key derived from filename pattern
    report_id       TEXT PRIMARY KEY,       -- "2026-01-15_한화투자증권_S-Oil"

    -- From df_firm.csv
    company         TEXT NOT NULL,          -- S-Oil
    title           TEXT,                   -- "2026년에도 정제마진 강세 지속"
    issuer          TEXT NOT NULL,          -- 한화투자증권 (broker)
    issue_date      DATE NOT NULL,          -- 2026-01-15
    viewer_count    INTEGER,                -- 1155
    summary         TEXT,                   -- Report summary
    pdf_link        TEXT,                   -- Naver PDF URL

    -- From with_author.csv
    author          TEXT,                   -- 이상헌
    author_email    TEXT,                   -- value3@hi-ib.com

    -- File references
    local_md_path   TEXT,                   -- /mnt/nas/gpt/Naver/Marker/...
    json_path       TEXT,                   -- /mnt/nas/gpt/Naver/Extraction_FirmAnalysis/...
    gemini_file_uri TEXT,                   -- gemini://file/xxx

    -- Status tracking
    has_extraction  BOOLEAN DEFAULT FALSE,
    has_md_file     BOOLEAN DEFAULT FALSE,
    gemini_uploaded BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ===========================================
-- Extraction Details Table
-- ===========================================
-- Source: Extraction JSON files
CREATE TABLE firm_extraction (
    report_id               TEXT PRIMARY KEY REFERENCES firm_reports(report_id) ON DELETE CASCADE,

    -- Metadata from JSON
    ticker                  TEXT,               -- 005930
    sector                  TEXT,               -- 반도체
    industry                TEXT,               -- 메모리 반도체, 파운드리
    extraction_confidence   FLOAT,              -- 0.95

    -- Valuation context
    valuation_regime        TEXT,               -- DEEP_DISCOUNT, FAIR, PREMIUM, etc.
    current_per             FLOAT,
    current_pbr             FLOAT,
    current_ev_ebitda       FLOAT,
    current_ps              FLOAT,
    dividend_yield          FLOAT,
    target_per              FLOAT,
    target_pbr              FLOAT,
    target_price            FLOAT,
    rating                  TEXT,               -- BUY, HOLD, SELL

    -- Historical valuation
    per_3y_avg              FLOAT,
    per_5y_avg              FLOAT,
    pbr_3y_avg              FLOAT,
    pbr_5y_avg              FLOAT,

    -- Earnings quality context
    roe_current             FLOAT,
    roe_3y_avg              FLOAT,
    roe_trend               TEXT,               -- IMPROVING, STABLE, DECLINING
    roa_current             FLOAT,
    roic_current            FLOAT,
    operating_margin        FLOAT,
    margin_trend            TEXT,               -- IMPROVING, STABLE, DECLINING
    profitability_regime    TEXT,               -- EXPANDING, STABLE, COMPRESSED

    -- Growth context
    revenue_growth_yoy      FLOAT,
    revenue_growth_3y_cagr  FLOAT,
    earnings_growth_yoy     FLOAT,
    growth_stage            TEXT,               -- EARLY, GROWTH, MATURE, DECLINE
    growth_regime           TEXT,               -- ACCELERATING, STABLE, DECELERATING, CYCLICAL_RECOVERY
    growth_drivers          TEXT[],             -- Array of growth drivers

    -- Cycle positioning
    business_cycle_stage    TEXT,               -- EARLY_CYCLE, MID_CYCLE, LATE_CYCLE
    industry_cycle_stage    TEXT,               -- EARLY_CYCLE, MID_CYCLE, LATE_CYCLE, DOWNCYCLE

    -- Rerating catalyst
    rerating_catalyst       TEXT,               -- Key catalyst for rerating

    -- Regime drivers and risk factors
    regime_drivers          TEXT[],             -- Array of regime drivers

    -- Raw JSON for complex queries
    raw_json                JSONB,

    -- Timestamps
    created_at              TIMESTAMP DEFAULT NOW(),
    updated_at              TIMESTAMP DEFAULT NOW()
);

-- ===========================================
-- Indexes for Performance
-- ===========================================

-- firm_reports indexes
CREATE INDEX idx_firm_reports_date ON firm_reports(issue_date DESC);
CREATE INDEX idx_firm_reports_issuer ON firm_reports(issuer);
CREATE INDEX idx_firm_reports_company ON firm_reports(company);
CREATE INDEX idx_firm_reports_author ON firm_reports(author);
CREATE INDEX idx_firm_reports_date_company ON firm_reports(issue_date DESC, company);

-- Full-text search on summary
CREATE INDEX idx_firm_reports_summary_fts ON firm_reports
USING GIN (to_tsvector('simple', coalesce(summary, '')));

-- Full-text search on title
CREATE INDEX idx_firm_reports_title_fts ON firm_reports
USING GIN (to_tsvector('simple', coalesce(title, '')));

-- firm_extraction indexes
CREATE INDEX idx_firm_extraction_ticker ON firm_extraction(ticker);
CREATE INDEX idx_firm_extraction_sector ON firm_extraction(sector);
CREATE INDEX idx_firm_extraction_valuation ON firm_extraction(valuation_regime);
CREATE INDEX idx_firm_extraction_growth ON firm_extraction(growth_regime);
CREATE INDEX idx_firm_extraction_cycle ON firm_extraction(industry_cycle_stage);
CREATE INDEX idx_firm_extraction_confidence ON firm_extraction(extraction_confidence DESC);

-- JSONB index for complex queries
CREATE INDEX idx_firm_extraction_raw_json ON firm_extraction USING GIN (raw_json);

-- Growth drivers array index
CREATE INDEX idx_firm_extraction_drivers ON firm_extraction USING GIN (growth_drivers);

-- ===========================================
-- Useful Views
-- ===========================================

-- Recent reports with extraction data
CREATE OR REPLACE VIEW v_firm_reports_full AS
SELECT
    r.report_id,
    r.company,
    r.title,
    r.issuer,
    r.issue_date,
    r.author,
    r.summary,
    r.viewer_count,
    e.ticker,
    e.sector,
    e.valuation_regime,
    e.current_per,
    e.current_pbr,
    e.roe_current,
    e.roe_trend,
    e.growth_regime,
    e.industry_cycle_stage,
    e.growth_drivers,
    e.extraction_confidence,
    r.has_extraction,
    r.has_md_file,
    r.gemini_uploaded
FROM firm_reports r
LEFT JOIN firm_extraction e ON r.report_id = e.report_id;

-- Broker coverage summary
CREATE OR REPLACE VIEW v_broker_coverage AS
SELECT
    issuer AS broker,
    COUNT(*) AS report_count,
    COUNT(DISTINCT company) AS companies_covered,
    MIN(issue_date) AS first_report,
    MAX(issue_date) AS last_report,
    COUNT(DISTINCT author) AS analyst_count
FROM firm_reports
GROUP BY issuer
ORDER BY report_count DESC;

-- Company coverage summary
CREATE OR REPLACE VIEW v_company_coverage AS
SELECT
    company,
    COUNT(*) AS report_count,
    COUNT(DISTINCT issuer) AS broker_count,
    MIN(issue_date) AS first_report,
    MAX(issue_date) AS last_report,
    array_agg(DISTINCT issuer) AS brokers
FROM firm_reports
GROUP BY company
ORDER BY report_count DESC;

-- Valuation regime distribution
CREATE OR REPLACE VIEW v_valuation_distribution AS
SELECT
    e.valuation_regime,
    COUNT(*) AS count,
    ROUND(AVG(e.current_per)::numeric, 2) AS avg_per,
    ROUND(AVG(e.current_pbr)::numeric, 2) AS avg_pbr,
    ROUND(AVG(e.roe_current)::numeric, 2) AS avg_roe
FROM firm_extraction e
WHERE e.valuation_regime IS NOT NULL
GROUP BY e.valuation_regime
ORDER BY count DESC;

-- ===========================================
-- Helper Functions
-- ===========================================

-- Function to generate report_id from components
CREATE OR REPLACE FUNCTION make_report_id(
    p_issue_date DATE,
    p_issuer TEXT,
    p_company TEXT
) RETURNS TEXT AS $$
BEGIN
    RETURN p_issue_date::TEXT || '_' || p_issuer || '_' || p_company;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_firm_reports_updated_at
    BEFORE UPDATE ON firm_reports
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_firm_extraction_updated_at
    BEFORE UPDATE ON firm_extraction
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- Sample Queries (Comments)
-- ===========================================

-- Find undervalued semiconductor stocks:
-- SELECT * FROM v_firm_reports_full
-- WHERE valuation_regime = 'DEEP_DISCOUNT' AND sector = '반도체'
-- ORDER BY issue_date DESC;

-- Get broker's recent coverage:
-- SELECT * FROM firm_reports
-- WHERE issuer = '한화투자증권' AND issue_date >= CURRENT_DATE - INTERVAL '30 days'
-- ORDER BY issue_date DESC;

-- Full-text search on summary:
-- SELECT * FROM firm_reports
-- WHERE to_tsvector('simple', summary) @@ to_tsquery('simple', 'HBM & 수혜')
-- ORDER BY issue_date DESC;

-- Companies with growth drivers containing 'AI':
-- SELECT DISTINCT company, growth_drivers
-- FROM firm_extraction
-- WHERE 'AI' = ANY(growth_drivers);

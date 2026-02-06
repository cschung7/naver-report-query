# SmartQuery Deployment Guide

## Step 1: Create GitHub Repo

Go to https://github.com/new and create:
- Name: `WWAI-SmartQuery`
- Visibility: Private

Then push:
```bash
cd /mnt/nas/WWAI/SmartQuery
git push -u origin main
```

## Step 2: Deploy to Railway

1. Go to https://railway.app/new
2. Click "Deploy from GitHub repo"
3. Select `cschung7/WWAI-SmartQuery`
4. Add PostgreSQL plugin (or use existing)

## Step 3: Set Environment Variables

In Railway dashboard, set these variables:

```
# PostgreSQL (use Railway-provided DATABASE_URL or set individually)
POSTGRES_URI_FIRM=postgresql://USER:PASS@HOST:PORT/naver_report
POSTGRES_URI_ECON=postgresql://USER:PASS@HOST:PORT/naver_econ
POSTGRES_URI_INDUSTRY=postgresql://USER:PASS@HOST:PORT/naver_industry
POSTGRES_URI_INVEST=postgresql://USER:PASS@HOST:PORT/naver_report

# Gemini AI (for summarization)
GEMINI_API_KEY=<your-key>

# Neo4j (set to 'none' to disable)
NEO4J_URI=none

# Railway auto-sets PORT
```

## Step 4: Migrate PostgreSQL Data

```bash
# Get Railway PostgreSQL connection string from dashboard
# Then restore each database:

psql $RAILWAY_PG_URL -c "CREATE DATABASE naver_report;"
psql $RAILWAY_PG_URL -c "CREATE DATABASE naver_econ;"
psql $RAILWAY_PG_URL -c "CREATE DATABASE naver_industry;"

psql $RAILWAY_PG_URL/naver_report < db_dumps/naver_report.sql
psql $RAILWAY_PG_URL/naver_econ < db_dumps/naver_econ.sql
psql $RAILWAY_PG_URL/naver_industry < db_dumps/naver_industry.sql
```

## Step 5: Verify

```bash
curl https://<railway-url>/health
curl https://<railway-url>/api
```

## Architecture

```
Railway Service (single)
+-- Gateway Flask ($PORT) <-- public
+-- FirmAnalysis Flask (:8080) <-- internal
+-- InvestmentStrategy Flask (:8001) <-- internal
+-- Industry Flask (:8002) <-- internal
+-- EconAnalysis Flask (:8003) <-- internal
```

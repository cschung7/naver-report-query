"""
Gemini Client for Research Summarization (via OpenRouter)

Generates research summaries from search results using Google's Gemini via OpenRouter API.
"""
import os
import requests
from typing import Dict, List, Any, Optional

# Load .env file if exists
try:
    from dotenv import load_dotenv
    # Try local .env first, then NAS fallback
    load_dotenv()
    if os.path.exists("/mnt/nas/gpt/.env"):
        load_dotenv("/mnt/nas/gpt/.env")
except ImportError:
    pass


class GeminiClient:
    """Client for generating research summaries using Gemini via OpenRouter."""

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        # Model configuration - use Gemini 2.5 Flash via OpenRouter
        self.model_name = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")

        # Get OpenRouter API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")

    def summarize_research(
        self,
        query: str,
        reports: List[Dict[str, Any]],
        claims: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 300
    ) -> Dict[str, Any]:
        """
        Generate a research summary from search results.

        Args:
            query: Original user query
            reports: List of report dictionaries with title, summary, company, etc.
            claims: Optional list of claims/insights from Neo4j
            max_tokens: Maximum tokens for response

        Returns:
            Dictionary with summary, key_findings, and references
        """
        # Build context from reports
        report_context = self._build_report_context(reports[:10])  # Top 10 reports
        claim_context = self._build_claim_context(claims[:20]) if claims else ""

        prompt = f"""[질문] {query}

[리포트] {report_context[:1500]}

반드시 아래 형식으로만 답변. 설명 없이 개조식으로만:

▶ 핵심: (20자 이내)
• (핵심1)
• (핵심2)
• (핵심3)
⚠ 리스크: (10자)
💡 시사점: (15자)"""

        try:
            response = requests.post(
                self.OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

            summary_text = result["choices"][0]["message"]["content"]

            # Extract references from reports
            references = self._extract_references(reports[:10])

            return {
                "success": True,
                "summary": summary_text,
                "references": references,
                "model": self.model_name,
                "reports_used": len(reports[:10])
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": None,
                "references": []
            }

    def _build_report_context(self, reports: List[Dict[str, Any]]) -> str:
        """Build context string from reports."""
        parts = []
        for i, r in enumerate(reports, 1):
            company = r.get('company', 'Unknown')
            issuer = r.get('issuer', 'Unknown')
            date = str(r.get('issue_date', r.get('date', '')))[:10]
            title = r.get('title', '')
            summary = r.get('summary', '')[:300] if r.get('summary') else ''

            part = f"[{i}] {company} - {title}\n"
            part += f"    발행: {issuer} ({date})\n"
            if summary:
                part += f"    요약: {summary}\n"
            parts.append(part)

        return "\n".join(parts)

    def _build_claim_context(self, claims: List[Dict[str, Any]]) -> str:
        """Build context string from claims."""
        if not claims:
            return ""

        parts = []
        for claim in claims[:15]:
            claim_type = claim.get('type', claim.get('claim_type', 'INSIGHT'))
            text = claim.get('text', claim.get('claim_text', ''))[:200]
            if text:
                parts.append(f"- [{claim_type}] {text}")

        return "\n".join(parts)

    def _extract_references(self, reports: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract reference information from reports."""
        refs = []
        for r in reports:
            refs.append({
                "company": r.get('company', 'Unknown'),
                "issuer": r.get('issuer', 'Unknown'),
                "date": str(r.get('issue_date', r.get('date', '')))[:10],
                "title": r.get('title', '')[:100],
                "report_id": r.get('report_id', r.get('id', ''))
            })
        return refs


# CLI for testing
if __name__ == "__main__":
    client = GeminiClient()

    # Test with sample data
    test_reports = [
        {
            "company": "삼성전자",
            "issuer": "키움증권",
            "issue_date": "2024-01-15",
            "title": "HBM 수요 급증으로 실적 개선 전망",
            "summary": "AI 서버 수요 증가로 HBM 매출이 크게 늘어날 것으로 예상됨."
        }
    ]

    result = client.summarize_research(
        query="삼성전자 HBM 전망",
        reports=test_reports
    )

    print("Summary Result:")
    print(result)

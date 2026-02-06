"""
Gemini Client for Research Summarization

Generates research summaries using:
  1. Native Google Gemini API (primary)
  2. OpenRouter API (fallback)
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
    """Client for generating research summaries using Gemini (native or OpenRouter)."""

    NATIVE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        # Try native Gemini API key first
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # Then OpenRouter as fallback
        self.openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.gemini_api_key and not self.openrouter_api_key:
            raise ValueError("Neither GEMINI_API_KEY nor OPENROUTER_API_KEY configured")

        # Model configuration
        self.native_model = os.getenv("GEMINI_MODEL_NATIVE", "gemini-2.5-flash")
        self.openrouter_model = os.getenv("GEMINI_MODEL_OPENROUTER", "google/gemini-3-flash-preview")

        backend = "native Gemini" if self.gemini_api_key else "OpenRouter"
        print(f"  GeminiClient initialized (primary: {backend})")

    def _call_native_gemini(self, prompt: str, max_tokens: int) -> str:
        """Call native Google Gemini API."""
        url = self.NATIVE_GEMINI_URL.format(model=self.native_model)
        # Disable thinking for summarization — it's simple extraction, not reasoning.
        # This cuts response time from ~4s to ~1-2s.
        response = requests.post(
            url,
            params={"key": self.gemini_api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max(max_tokens, 1024),
                    "temperature": 0.3,
                    "thinkingConfig": {"thinkingBudget": 0}
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        parts = result["candidates"][0]["content"]["parts"]
        for part in reversed(parts):
            if "text" in part:
                return part["text"]
        return parts[0]["text"]

    def _call_openrouter(self, prompt: str, max_tokens: int) -> str:
        """Call OpenRouter API (fallback)."""
        response = requests.post(
            self.OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def summarize_research(
        self,
        query: str,
        reports: List[Dict[str, Any]],
        claims: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 300
    ) -> Dict[str, Any]:
        """
        Generate a research summary from search results.

        Tries native Gemini first, falls back to OpenRouter on failure.
        """
        # Build context from reports
        report_context = self._build_report_context(reports[:10])

        prompt = f"""[질문] {query}

[리포트] {report_context[:1500]}

반드시 아래 형식으로만 답변. 설명 없이 개조식으로만:

▶ 핵심: (20자 이내)
• (핵심1)
• (핵심2)
• (핵심3)
⚠ 리스크: (10자)
💡 시사점: (15자)"""

        used_model = None
        summary_text = None
        last_error = None

        # Try native Gemini first
        if self.gemini_api_key:
            try:
                summary_text = self._call_native_gemini(prompt, max_tokens)
                used_model = f"native:{self.native_model}"
            except Exception as e:
                last_error = e
                print(f"  Native Gemini failed: {e}, trying OpenRouter...")

        # Fallback to OpenRouter
        if summary_text is None and self.openrouter_api_key:
            try:
                summary_text = self._call_openrouter(prompt, max_tokens)
                used_model = f"openrouter:{self.openrouter_model}"
            except Exception as e:
                last_error = e
                print(f"  OpenRouter also failed: {e}")

        if summary_text is None:
            return {
                "success": False,
                "error": str(last_error) if last_error else "No API key available",
                "summary": None,
                "references": []
            }

        references = self._extract_references(reports[:10])
        return {
            "success": True,
            "summary": summary_text,
            "references": references,
            "model": used_model,
            "reports_used": len(reports[:10])
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

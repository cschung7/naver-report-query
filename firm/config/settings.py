"""
NaverReport-FirmAnalysis Configuration Settings

Central configuration for all data paths, database connections, and system settings.
"""
import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # ===========================================
    # Database Connections
    # ===========================================
    postgres_uri: str = field(
        default_factory=lambda: os.getenv(
            "POSTGRES_URI_FIRM",
            os.getenv("POSTGRES_URI", "postgresql://kg_user:kg_secure_password_2025@localhost:5432/naver_report")
        )
    )
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password")
    )

    # ===========================================
    # API Keys
    # ===========================================
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )

    # ===========================================
    # Data Paths (Source)
    # ===========================================
    # df_files - Source of Truth for metadata
    df_files_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DF_FILES_PATH", "/mnt/nas/gpt/Naver/Job/df_files")
        )
    )

    # Extraction JSON files
    extraction_firm_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "EXTRACTION_FIRM_PATH",
                "/mnt/nas/gpt/Naver/Extraction_FirmAnalysis/extractions"
            )
        )
    )

    # Marker MD files
    marker_firm_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MARKER_FIRM_PATH", "/mnt/nas/gpt/Naver/Marker/FirmAnalysis")
        )
    )

    # ===========================================
    # Data File Patterns
    # ===========================================
    # Latest df_firm file pattern
    df_firm_pattern: str = "df_firm_*.csv"
    df_firm_with_author_pattern: str = "df_*_firmAnalysis_with_author_*.csv"

    # ===========================================
    # LRS Integration (Optional)
    # ===========================================
    lrs_data_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "LRS_DATA_PATH",
                "/mnt/nas/AutoGluon/AutoML_Krx/Filter/UCS_LRS"
            )
        )
    )

    # ===========================================
    # Processing Settings
    # ===========================================
    batch_size: int = 1000
    gemini_upload_months: int = 6  # Upload reports from last N months
    max_gemini_files: int = 10000  # Maximum files to upload to Gemini

    # ===========================================
    # Derived Paths
    # ===========================================
    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return Path(__file__).parent.parent

    @property
    def schema_path(self) -> Path:
        """Path to SQL schema file."""
        return self.project_root / "config" / "schema.sql"

    # ===========================================
    # Helper Methods
    # ===========================================
    def get_latest_df_firm(self) -> Optional[Path]:
        """Get the latest df_firm CSV file."""
        files = sorted(
            self.df_files_path.glob(self.df_firm_pattern),
            key=lambda x: x.stem,
            reverse=True
        )
        return files[0] if files else None

    def get_latest_df_firm_with_author(self) -> Optional[Path]:
        """Get the latest df_firm_with_author CSV file."""
        files = sorted(
            self.df_files_path.glob(self.df_firm_with_author_pattern),
            key=lambda x: x.stem,
            reverse=True
        )
        return files[0] if files else None

    def validate(self) -> list[str]:
        """Validate settings and return list of errors."""
        errors = []

        # Check data paths exist
        if not self.df_files_path.exists():
            errors.append(f"df_files_path does not exist: {self.df_files_path}")

        if not self.extraction_firm_path.exists():
            errors.append(
                f"extraction_firm_path does not exist: {self.extraction_firm_path}"
            )

        if not self.marker_firm_path.exists():
            errors.append(f"marker_firm_path does not exist: {self.marker_firm_path}")

        # Check for required API keys
        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY not set (required for semantic search)")

        return errors


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()

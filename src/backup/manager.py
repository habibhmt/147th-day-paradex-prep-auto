"""Backup and export management for trading bot."""

import json
import logging
import os
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""

    FULL = "full"  # Everything
    TRADES = "trades"  # Trade history only
    CONFIG = "config"  # Configuration only
    ACCOUNTS = "accounts"  # Account data only
    ANALYTICS = "analytics"  # Analytics data only


class ExportFormat(Enum):
    """Export file formats."""

    JSON = "json"
    CSV = "csv"
    TAR_GZ = "tar.gz"


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    backup_path: Optional[str] = None
    backup_type: BackupType = BackupType.FULL
    size_bytes: int = 0
    files_count: int = 0
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "backup_path": self.backup_path,
            "backup_type": self.backup_type.value,
            "size_bytes": self.size_bytes,
            "files_count": self.files_count,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class BackupConfig:
    """Configuration for backup operations."""

    backup_dir: str = "backups"
    max_backups: int = 10  # Keep last N backups
    compress: bool = True
    include_logs: bool = False
    auto_backup_interval: int = 86400  # Daily (in seconds)
    encrypt: bool = False  # Future feature

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "backup_dir": self.backup_dir,
            "max_backups": self.max_backups,
            "compress": self.compress,
            "include_logs": self.include_logs,
            "auto_backup_interval": self.auto_backup_interval,
            "encrypt": self.encrypt,
        }


@dataclass
class BackupManager:
    """Manages backup and export operations.

    Features:
    - Full and partial backups
    - Export to JSON/CSV
    - Automatic backup rotation
    - Compression support
    - Restore functionality
    """

    config: BackupConfig = field(default_factory=BackupConfig)
    _backup_history: List[BackupResult] = field(default_factory=list)
    _last_backup_time: float = 0.0

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._backup_history = []
        self._last_backup_time = 0.0
        self._ensure_backup_dir()

    def _ensure_backup_dir(self) -> None:
        """Ensure backup directory exists."""
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

    def _generate_backup_name(self, backup_type: BackupType) -> str:
        """Generate unique backup filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{backup_type.value}_{timestamp}"

    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        data: Optional[Dict[str, Any]] = None,
    ) -> BackupResult:
        """Create a backup.

        Args:
            backup_type: Type of backup to create
            data: Data to backup (if None, will collect from sources)

        Returns:
            BackupResult with status and path
        """
        start_time = time.time()

        try:
            backup_name = self._generate_backup_name(backup_type)
            backup_path = Path(self.config.backup_dir) / backup_name

            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Collect data if not provided
            if data is None:
                data = self._collect_backup_data(backup_type)

            # Write data files
            files_count = self._write_backup_files(backup_path, data)

            # Compress if enabled
            final_path = str(backup_path)
            if self.config.compress:
                final_path = self._compress_backup(backup_path)
                # Remove uncompressed directory
                shutil.rmtree(backup_path)

            # Calculate size
            if os.path.isfile(final_path):
                size_bytes = os.path.getsize(final_path)
            else:
                size_bytes = sum(
                    f.stat().st_size for f in Path(final_path).rglob("*") if f.is_file()
                )

            # Update last backup time
            self._last_backup_time = time.time()

            # Create result
            result = BackupResult(
                success=True,
                backup_path=final_path,
                backup_type=backup_type,
                size_bytes=size_bytes,
                files_count=files_count,
                duration_ms=(time.time() - start_time) * 1000,
            )

            # Add to history
            self._backup_history.append(result)

            # Rotate old backups
            self._rotate_backups()

            logger.info(f"Backup created: {final_path} ({size_bytes} bytes)")
            return result

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return BackupResult(
                success=False,
                backup_type=backup_type,
                error_message=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _collect_backup_data(self, backup_type: BackupType) -> Dict[str, Any]:
        """Collect data for backup based on type."""
        data = {
            "metadata": {
                "backup_type": backup_type.value,
                "timestamp": time.time(),
                "version": "1.0.0",
            }
        }

        if backup_type in (BackupType.FULL, BackupType.TRADES):
            data["trades"] = []  # Would be populated from trade history DB

        if backup_type in (BackupType.FULL, BackupType.CONFIG):
            data["config"] = {}  # Would be populated from settings

        if backup_type in (BackupType.FULL, BackupType.ACCOUNTS):
            data["accounts"] = []  # Would be populated from account manager

        if backup_type in (BackupType.FULL, BackupType.ANALYTICS):
            data["analytics"] = {}  # Would be populated from analytics

        return data

    def _write_backup_files(
        self, backup_path: Path, data: Dict[str, Any]
    ) -> int:
        """Write backup data to files."""
        files_count = 0

        for key, value in data.items():
            file_path = backup_path / f"{key}.json"
            with open(file_path, "w") as f:
                json.dump(value, f, indent=2, default=str)
            files_count += 1

        return files_count

    def _compress_backup(self, backup_path: Path) -> str:
        """Compress backup directory to tar.gz."""
        archive_path = f"{backup_path}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)

        return archive_path

    def _rotate_backups(self) -> int:
        """Remove old backups beyond max_backups limit."""
        backup_dir = Path(self.config.backup_dir)
        backups = sorted(
            [f for f in backup_dir.iterdir() if f.name.startswith("backup_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        removed = 0
        for backup in backups[self.config.max_backups :]:
            if backup.is_file():
                backup.unlink()
            else:
                shutil.rmtree(backup)
            removed += 1
            logger.info(f"Removed old backup: {backup}")

        return removed

    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore data from a backup.

        Args:
            backup_path: Path to backup file or directory

        Returns:
            Restored data dictionary
        """
        path = Path(backup_path)

        if not path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        data = {}

        # Handle compressed backup
        if path.suffix == ".gz" and str(path).endswith(".tar.gz"):
            extract_dir = path.parent / path.stem.replace(".tar", "")
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(path.parent)
            path = extract_dir

        # Read all JSON files
        if path.is_dir():
            for json_file in path.glob("*.json"):
                key = json_file.stem
                with open(json_file) as f:
                    data[key] = json.load(f)

        return data

    def export_trades_csv(
        self,
        trades: List[Dict],
        output_path: str,
    ) -> int:
        """Export trades to CSV format.

        Args:
            trades: List of trade dictionaries
            output_path: Output file path

        Returns:
            Number of trades exported
        """
        if not trades:
            return 0

        fieldnames = list(trades[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)

        logger.info(f"Exported {len(trades)} trades to {output_path}")
        return len(trades)

    def export_analytics_json(
        self,
        analytics: Dict[str, Any],
        output_path: str,
    ) -> bool:
        """Export analytics data to JSON.

        Args:
            analytics: Analytics data dictionary
            output_path: Output file path

        Returns:
            Success status
        """
        try:
            with open(output_path, "w") as f:
                json.dump(analytics, f, indent=2, default=str)
            logger.info(f"Exported analytics to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export analytics: {e}")
            return False

    def export_report(
        self,
        data: Dict[str, Any],
        output_path: str,
        format_type: ExportFormat = ExportFormat.JSON,
    ) -> bool:
        """Export a report in specified format.

        Args:
            data: Report data
            output_path: Output file path
            format_type: Export format

        Returns:
            Success status
        """
        try:
            if format_type == ExportFormat.JSON:
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            elif format_type == ExportFormat.CSV:
                if "trades" in data:
                    self.export_trades_csv(data["trades"], output_path)
                else:
                    # Flatten data for CSV
                    with open(output_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        for key, value in data.items():
                            writer.writerow([key, str(value)])

            logger.info(f"Exported report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False

    def get_backup_list(self) -> List[Dict]:
        """Get list of available backups.

        Returns:
            List of backup info dictionaries
        """
        backup_dir = Path(self.config.backup_dir)
        backups = []

        if not backup_dir.exists():
            return backups

        for item in backup_dir.iterdir():
            if item.name.startswith("backup_"):
                info = {
                    "name": item.name,
                    "path": str(item),
                    "size_bytes": (
                        item.stat().st_size
                        if item.is_file()
                        else sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    ),
                    "created_at": item.stat().st_mtime,
                    "is_compressed": item.suffix == ".gz",
                }
                backups.append(info)

        # Sort by creation time, newest first
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups

    def get_backup_history(self) -> List[Dict]:
        """Get backup operation history.

        Returns:
            List of backup results
        """
        return [r.to_dict() for r in self._backup_history]

    def should_auto_backup(self) -> bool:
        """Check if automatic backup is due.

        Returns:
            True if backup should be triggered
        """
        if self._last_backup_time == 0:
            return True

        elapsed = time.time() - self._last_backup_time
        return elapsed >= self.config.auto_backup_interval

    def delete_backup(self, backup_path: str) -> bool:
        """Delete a specific backup.

        Args:
            backup_path: Path to backup to delete

        Returns:
            Success status
        """
        path = Path(backup_path)

        if not path.exists():
            return False

        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            logger.info(f"Deleted backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False

    def get_total_backup_size(self) -> int:
        """Get total size of all backups in bytes.

        Returns:
            Total size in bytes
        """
        backup_dir = Path(self.config.backup_dir)
        if not backup_dir.exists():
            return 0

        total = 0
        for item in backup_dir.iterdir():
            if item.name.startswith("backup_"):
                if item.is_file():
                    total += item.stat().st_size
                else:
                    total += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())

        return total

    def get_status(self) -> Dict:
        """Get backup manager status.

        Returns:
            Status dictionary
        """
        backups = self.get_backup_list()

        return {
            "config": self.config.to_dict(),
            "backup_count": len(backups),
            "total_size_bytes": self.get_total_backup_size(),
            "last_backup_time": self._last_backup_time,
            "auto_backup_due": self.should_auto_backup(),
            "recent_backups": backups[:5],
        }

    def cleanup_old_backups(self, days: int = 30) -> int:
        """Remove backups older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of backups removed
        """
        backup_dir = Path(self.config.backup_dir)
        if not backup_dir.exists():
            return 0

        cutoff_time = time.time() - (days * 86400)
        removed = 0

        for item in backup_dir.iterdir():
            if item.name.startswith("backup_"):
                if item.stat().st_mtime < cutoff_time:
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
                    removed += 1
                    logger.info(f"Cleaned up old backup: {item}")

        return removed

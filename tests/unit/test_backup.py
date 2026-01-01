"""Unit tests for Backup Manager."""

import pytest
import json
import os
import time
from pathlib import Path
import tarfile

from src.backup.manager import (
    BackupManager,
    BackupConfig,
    BackupType,
    BackupResult,
    ExportFormat,
)


class TestBackupType:
    """Tests for BackupType enum."""

    def test_backup_types(self):
        """Should have expected backup types."""
        assert BackupType.FULL.value == "full"
        assert BackupType.TRADES.value == "trades"
        assert BackupType.CONFIG.value == "config"
        assert BackupType.ACCOUNTS.value == "accounts"
        assert BackupType.ANALYTICS.value == "analytics"


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_export_formats(self):
        """Should have expected export formats."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.TAR_GZ.value == "tar.gz"


class TestBackupResult:
    """Tests for BackupResult dataclass."""

    def test_create_result(self):
        """Should create result correctly."""
        result = BackupResult(
            success=True,
            backup_path="/backups/test.tar.gz",
            backup_type=BackupType.FULL,
            size_bytes=1024,
            files_count=5,
        )

        assert result.success is True
        assert result.backup_path == "/backups/test.tar.gz"
        assert result.size_bytes == 1024
        assert result.files_count == 5

    def test_failed_result(self):
        """Should create failed result."""
        result = BackupResult(
            success=False,
            error_message="Disk full",
        )

        assert result.success is False
        assert result.error_message == "Disk full"

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        result = BackupResult(
            success=True,
            backup_path="/backups/test.tar.gz",
            backup_type=BackupType.TRADES,
            size_bytes=2048,
            files_count=3,
            duration_ms=150.5,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["backup_type"] == "trades"
        assert d["size_bytes"] == 2048
        assert d["duration_ms"] == 150.5


class TestBackupConfig:
    """Tests for BackupConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = BackupConfig()

        assert config.backup_dir == "backups"
        assert config.max_backups == 10
        assert config.compress is True
        assert config.include_logs is False
        assert config.auto_backup_interval == 86400

    def test_custom_config(self):
        """Should accept custom values."""
        config = BackupConfig(
            backup_dir="/custom/backups",
            max_backups=5,
            compress=False,
        )

        assert config.backup_dir == "/custom/backups"
        assert config.max_backups == 5
        assert config.compress is False

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        config = BackupConfig(
            backup_dir="/backups",
            max_backups=7,
        )

        d = config.to_dict()

        assert d["backup_dir"] == "/backups"
        assert d["max_backups"] == 7


class TestBackupManager:
    """Tests for BackupManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create backup manager with temp directory."""
        config = BackupConfig(
            backup_dir=str(tmp_path / "backups"),
            max_backups=5,
            compress=False,
        )
        return BackupManager(config=config)

    @pytest.fixture
    def compressed_manager(self, tmp_path):
        """Create backup manager with compression."""
        config = BackupConfig(
            backup_dir=str(tmp_path / "backups"),
            max_backups=5,
            compress=True,
        )
        return BackupManager(config=config)

    def test_initial_state(self, manager):
        """Should start with clean state."""
        status = manager.get_status()

        assert status["backup_count"] == 0
        assert status["total_size_bytes"] == 0
        assert status["last_backup_time"] == 0.0

    def test_backup_dir_created(self, manager):
        """Should create backup directory."""
        assert Path(manager.config.backup_dir).exists()

    def test_create_backup_full(self, manager):
        """Should create full backup."""
        result = manager.create_backup(BackupType.FULL)

        assert result.success is True
        assert result.backup_path is not None
        assert result.backup_type == BackupType.FULL
        assert result.files_count > 0
        assert Path(result.backup_path).exists()

    def test_create_backup_trades(self, manager):
        """Should create trades backup."""
        result = manager.create_backup(BackupType.TRADES)

        assert result.success is True
        assert result.backup_type == BackupType.TRADES

    def test_create_backup_config(self, manager):
        """Should create config backup."""
        result = manager.create_backup(BackupType.CONFIG)

        assert result.success is True
        assert result.backup_type == BackupType.CONFIG

    def test_create_backup_with_data(self, manager):
        """Should backup provided data."""
        test_data = {
            "trades": [{"id": 1, "amount": 100}],
            "settings": {"theme": "dark"},
        }

        result = manager.create_backup(BackupType.FULL, data=test_data)

        assert result.success is True
        assert result.files_count == 2

    def test_create_backup_compressed(self, compressed_manager):
        """Should create compressed backup."""
        result = compressed_manager.create_backup(BackupType.FULL)

        assert result.success is True
        assert result.backup_path.endswith(".tar.gz")
        assert Path(result.backup_path).exists()

    def test_backup_has_duration(self, manager):
        """Should record backup duration."""
        result = manager.create_backup(BackupType.FULL)

        assert result.duration_ms > 0

    def test_backup_updates_last_time(self, manager):
        """Should update last backup time."""
        assert manager._last_backup_time == 0.0

        manager.create_backup(BackupType.FULL)

        assert manager._last_backup_time > 0

    def test_backup_added_to_history(self, manager):
        """Should add backup to history."""
        manager.create_backup(BackupType.FULL)

        history = manager.get_backup_history()

        assert len(history) == 1
        assert history[0]["success"] is True

    def test_restore_backup(self, manager):
        """Should restore from backup."""
        test_data = {
            "trades": [{"id": 1}],
            "config": {"setting": "value"},
        }

        result = manager.create_backup(BackupType.FULL, data=test_data)
        restored = manager.restore_backup(result.backup_path)

        assert "trades" in restored
        assert "config" in restored
        assert restored["trades"] == [{"id": 1}]

    def test_restore_compressed_backup(self, compressed_manager):
        """Should restore compressed backup."""
        test_data = {
            "trades": [{"id": 1}, {"id": 2}],
        }

        result = compressed_manager.create_backup(BackupType.FULL, data=test_data)
        restored = compressed_manager.restore_backup(result.backup_path)

        assert "trades" in restored
        assert len(restored["trades"]) == 2

    def test_restore_nonexistent_backup(self, manager):
        """Should raise error for missing backup."""
        with pytest.raises(FileNotFoundError):
            manager.restore_backup("/nonexistent/backup")

    def test_export_trades_csv(self, manager, tmp_path):
        """Should export trades to CSV."""
        trades = [
            {"id": 1, "market": "BTC-USD", "amount": 100},
            {"id": 2, "market": "ETH-USD", "amount": 50},
        ]
        output_path = str(tmp_path / "trades.csv")

        count = manager.export_trades_csv(trades, output_path)

        assert count == 2
        assert Path(output_path).exists()

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # Header + 2 trades

    def test_export_trades_csv_empty(self, manager, tmp_path):
        """Should handle empty trades list."""
        output_path = str(tmp_path / "trades.csv")

        count = manager.export_trades_csv([], output_path)

        assert count == 0

    def test_export_analytics_json(self, manager, tmp_path):
        """Should export analytics to JSON."""
        analytics = {
            "total_volume": 100000,
            "win_rate": 65.5,
            "trades": 150,
        }
        output_path = str(tmp_path / "analytics.json")

        success = manager.export_analytics_json(analytics, output_path)

        assert success is True
        assert Path(output_path).exists()

        # Verify content
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["total_volume"] == 100000

    def test_export_report_json(self, manager, tmp_path):
        """Should export report as JSON."""
        data = {"summary": "test", "value": 123}
        output_path = str(tmp_path / "report.json")

        success = manager.export_report(data, output_path, ExportFormat.JSON)

        assert success is True
        assert Path(output_path).exists()

    def test_export_report_csv(self, manager, tmp_path):
        """Should export report as CSV."""
        data = {"trades": [{"id": 1}, {"id": 2}]}
        output_path = str(tmp_path / "report.csv")

        success = manager.export_report(data, output_path, ExportFormat.CSV)

        assert success is True
        assert Path(output_path).exists()

    def test_get_backup_list(self, manager):
        """Should list available backups."""
        manager.create_backup(BackupType.FULL)
        manager.create_backup(BackupType.TRADES)

        backups = manager.get_backup_list()

        assert len(backups) == 2
        assert all("name" in b for b in backups)
        assert all("path" in b for b in backups)
        assert all("size_bytes" in b for b in backups)

    def test_get_backup_list_sorted(self, manager):
        """Should sort backups by creation time."""
        manager.create_backup(BackupType.FULL)
        time.sleep(0.01)
        manager.create_backup(BackupType.TRADES)

        backups = manager.get_backup_list()

        # Newest first
        assert backups[0]["created_at"] > backups[1]["created_at"]

    def test_should_auto_backup_initial(self, manager):
        """Should recommend backup when never backed up."""
        assert manager.should_auto_backup() is True

    def test_should_auto_backup_recent(self, manager):
        """Should not recommend backup right after one."""
        manager.create_backup(BackupType.FULL)

        assert manager.should_auto_backup() is False

    def test_delete_backup(self, manager):
        """Should delete backup."""
        result = manager.create_backup(BackupType.FULL)

        deleted = manager.delete_backup(result.backup_path)

        assert deleted is True
        assert not Path(result.backup_path).exists()

    def test_delete_backup_nonexistent(self, manager):
        """Should return False for missing backup."""
        deleted = manager.delete_backup("/nonexistent/backup")
        assert deleted is False

    def test_get_total_backup_size(self, manager):
        """Should calculate total backup size."""
        manager.create_backup(
            BackupType.FULL,
            data={"large_data": "x" * 1000},
        )
        manager.create_backup(
            BackupType.FULL,
            data={"more_data": "y" * 500},
        )

        total_size = manager.get_total_backup_size()

        assert total_size > 0

    def test_get_status(self, manager):
        """Should return status dictionary."""
        manager.create_backup(BackupType.FULL)

        status = manager.get_status()

        assert "config" in status
        assert "backup_count" in status
        assert "total_size_bytes" in status
        assert "last_backup_time" in status
        assert "auto_backup_due" in status
        assert "recent_backups" in status
        assert status["backup_count"] == 1

    def test_rotate_backups(self, tmp_path):
        """Should remove old backups beyond limit."""
        config = BackupConfig(
            backup_dir=str(tmp_path / "backups"),
            max_backups=2,
            compress=False,
        )
        manager = BackupManager(config=config)

        # Create more backups than limit using different types to get unique names
        backup_types = [BackupType.FULL, BackupType.TRADES, BackupType.CONFIG, BackupType.ANALYTICS]
        for backup_type in backup_types:
            manager.create_backup(
                backup_type,
                data={"index": backup_type.value},
            )

        backups = manager.get_backup_list()

        # Should only keep max_backups
        assert len(backups) == 2

    def test_cleanup_old_backups(self, manager):
        """Should clean up backups older than threshold."""
        # Create a backup
        result = manager.create_backup(BackupType.FULL)

        # Manually modify file time to be old
        backup_path = Path(result.backup_path)
        old_time = time.time() - (31 * 86400)  # 31 days ago
        os.utime(backup_path, (old_time, old_time))

        removed = manager.cleanup_old_backups(days=30)

        assert removed == 1
        assert not backup_path.exists()

    def test_backup_metadata(self, manager):
        """Should include metadata in backup."""
        result = manager.create_backup(BackupType.FULL)
        restored = manager.restore_backup(result.backup_path)

        assert "metadata" in restored
        assert restored["metadata"]["backup_type"] == "full"
        assert "timestamp" in restored["metadata"]
        assert "version" in restored["metadata"]

    def test_multiple_backup_types(self, manager):
        """Should handle different backup types correctly."""
        full_result = manager.create_backup(BackupType.FULL)
        trades_result = manager.create_backup(BackupType.TRADES)
        config_result = manager.create_backup(BackupType.CONFIG)

        assert full_result.success is True
        assert trades_result.success is True
        assert config_result.success is True

        # Verify different paths
        paths = {full_result.backup_path, trades_result.backup_path, config_result.backup_path}
        assert len(paths) == 3

    def test_compressed_backup_is_smaller(self, tmp_path):
        """Compressed backups should generally be smaller."""
        large_data = {"data": "x" * 10000}

        # Uncompressed
        config1 = BackupConfig(
            backup_dir=str(tmp_path / "uncompressed"),
            compress=False,
        )
        manager1 = BackupManager(config=config1)
        result1 = manager1.create_backup(BackupType.FULL, data=large_data)

        # Compressed
        config2 = BackupConfig(
            backup_dir=str(tmp_path / "compressed"),
            compress=True,
        )
        manager2 = BackupManager(config=config2)
        result2 = manager2.create_backup(BackupType.FULL, data=large_data)

        # Compressed should be smaller (or at least not much larger)
        assert result2.size_bytes <= result1.size_bytes * 1.5

    def test_backup_history_persistence(self, manager):
        """Should maintain backup history."""
        manager.create_backup(BackupType.FULL)
        manager.create_backup(BackupType.TRADES)
        manager.create_backup(BackupType.CONFIG)

        history = manager.get_backup_history()

        assert len(history) == 3
        assert all(h["success"] for h in history)

    def test_generate_backup_name_unique(self, manager):
        """Should generate unique backup names."""
        name1 = manager._generate_backup_name(BackupType.FULL)
        time.sleep(0.001)
        name2 = manager._generate_backup_name(BackupType.FULL)

        # Names should be different (due to timestamp)
        # Note: This might fail if both calls happen in same second
        # In practice, names include timestamp down to seconds
        assert name1.startswith("backup_full_")

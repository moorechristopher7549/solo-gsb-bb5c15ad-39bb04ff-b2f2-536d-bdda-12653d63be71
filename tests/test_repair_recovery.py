"""启动时对原位修复崩溃备份（.repairbak）的恢复。

进程在“旧分片改名 .repairbak、备份清理前”退出后重启：
  * 服务接收请求前扫描所有修复备份；
  * 仅对 uploading 会话且区间与 chunks 表精确一致的备份，在会话锁下
    用旧备份恢复正式分片——正式路径缺失，或已落入未确认的新文件，
    一律以旧备份为准；恢复成功后备份消失；
  * 恢复失败保留备份并使启动报错；
  * 后台定时清理不得删除仍需恢复的备份。
"""
from __future__ import annotations

import hashlib
import os
import time as _time

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.recovery import RepairRecoveryError, recover_pending_repairs


def _create(client: TestClient, data: bytes, name: str = "data.bin") -> str:
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _put(client: TestClient, sid: str, start: int, end: int, content: bytes):
    return client.put(
        f"/api/v1/sessions/{sid}/chunks?start={start}&end={end}",
        content=content,
    )


def _complete_ok(client: TestClient, sid: str, data: bytes) -> None:
    r = client.post(f"/api/v1/sessions/{sid}/complete")
    assert r.status_code == 200, r.text
    dl = client.get(f"/api/v1/sessions/{sid}/download")
    assert dl.status_code == 200
    assert dl.content == data


# ---------------------------------------------------------------------------
# 恢复：以旧备份为准
# ---------------------------------------------------------------------------


def test_recovery_restores_backup_over_unconfirmed_new_file(client: TestClient) -> None:
    # 崩溃窗口：旧分片已改名备份，新分片已落位但事务未确认。
    # 重启后必须以旧备份为准，覆盖同路径的未确认新文件。
    data = b"old-content-" + bytes(range(40))
    sid = _create(client, data)
    _put(client, sid, 0, len(data), data).raise_for_status()

    final = storage.part_path(sid, 0, len(data))
    backup = storage.part_repair_backup_path(sid, 0, len(data))
    # 模拟崩溃瞬间：正式分片改名备份，路径上落入未确认的新内容。
    final.replace(backup)
    final.write_bytes(b"!" * len(data))  # 等长的未确认新文件
    assert backup.read_bytes() == data

    recover_pending_repairs()

    assert final.read_bytes() == data
    assert not backup.exists()
    # 恢复后会话可直接用旧内容完成整文件校验，无需客户端任何干预。
    _complete_ok(client, sid, data)


def test_recovery_restores_backup_when_final_missing(client: TestClient) -> None:
    # 崩溃窗口更早：旧分片已改名备份，新分片尚未落位，正式路径缺文件。
    data = bytes((i * 5 + 1) % 256 for i in range(1200))
    sid = _create(client, data)
    _put(client, sid, 0, 600, data[:600]).raise_for_status()
    _put(client, sid, 600, 1200, data[600:]).raise_for_status()

    final = storage.part_path(sid, 600, 1200)
    backup = storage.part_repair_backup_path(sid, 600, 1200)
    final.replace(backup)
    assert not final.exists()
    assert backup.read_bytes() == data[600:]

    recover_pending_repairs()

    assert final.read_bytes() == data[600:]
    assert not backup.exists()
    # 相邻分片不受影响。
    assert storage.part_path(sid, 0, 600).read_bytes() == data[:600]
    _complete_ok(client, sid, data)


def test_recovery_multiple_chunks_and_sessions(client: TestClient) -> None:
    data_a = bytes((i * 3) % 256 for i in range(900))
    data_b = bytes((i * 7) % 256 for i in range(900))
    sid_a = _create(client, data_a)
    sid_b = _create(client, data_b)
    for sid, data in ((sid_a, data_a), (sid_b, data_b)):
        _put(client, sid, 0, 300, data[:300]).raise_for_status()
        _put(client, sid, 300, 600, data[300:600]).raise_for_status()
        _put(client, sid, 600, 900, data[600:]).raise_for_status()

    cases = [
        (sid_a, 0, 300, data_a[:300]),
        (sid_a, 600, 900, data_a[600:]),
        (sid_b, 300, 600, data_b[300:600]),
    ]
    for sid, start, end, old in cases:
        final = storage.part_path(sid, start, end)
        final.replace(storage.part_repair_backup_path(sid, start, end))
        final.write_bytes(b"?" * (end - start))

    recover_pending_repairs()

    for sid, start, end, old in cases:
        assert storage.part_path(sid, start, end).read_bytes() == old
        assert not storage.part_repair_backup_path(sid, start, end).exists()
    # 未崩溃的分片保持不动；两个会话都能完成。
    assert storage.part_path(sid_a, 300, 600).read_bytes() == data_a[300:600]
    _complete_ok(client, sid_a, data_a)
    _complete_ok(client, sid_b, data_b)


# ---------------------------------------------------------------------------
# 不恢复的备份：会话非 uploading / 区间无精确登记
# ---------------------------------------------------------------------------


def test_recovery_ignores_backup_without_exact_registered_range(
    client: TestClient,
) -> None:
    data = b"x" * 500
    sid = _create(client, data)
    _put(client, sid, 0, 500, data).raise_for_status()

    # 备份区间 [0, 100) 在 chunks 表中没有精确登记（只有 [0, 500)）：
    # 不是崩溃修复的产物，恢复不得凭空造出未登记分片；备份原样保留。
    bogus_backup = storage.part_repair_backup_path(sid, 0, 100)
    bogus_backup.write_bytes(b"bogus")
    recover_pending_repairs()
    assert bogus_backup.exists()
    assert storage.part_path(sid, 0, 500).read_bytes() == data


def test_recovery_ignores_backups_of_non_uploading_sessions(
    client: TestClient,
) -> None:
    # completed 会话目录里（极端情况下）残留备份：不恢复、不报错，
    # 由 sweep_completed_parts 与定时清理回收。
    data = b"f" * 200
    sid = _create(client, data)
    _put(client, sid, 0, 200, data).raise_for_status()
    _complete_ok(client, sid, data)

    backup_dir = storage.parts_dir(sid)
    backup_dir.mkdir(parents=True, exist_ok=True)
    leftover = storage.part_repair_backup_path(sid, 0, 200)
    leftover.write_bytes(data)
    recover_pending_repairs()  # 不抛错
    assert leftover.exists()
    # 成品不受影响。
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data


# ---------------------------------------------------------------------------
# 恢复失败：保留备份并使启动报错
# ---------------------------------------------------------------------------


def test_recovery_failure_keeps_backup_and_raises(
    client: TestClient, monkeypatch
) -> None:
    data = b"r" * 120
    sid = _create(client, data)
    _put(client, sid, 0, 120, data).raise_for_status()

    final = storage.part_path(sid, 0, 120)
    backup = storage.part_repair_backup_path(sid, 0, 120)
    final.replace(backup)
    assert not final.exists()

    # 注入恢复期间的 IO 故障（如目标目录不可写）：恢复必须失败、
    # 备份保留，并抛出 RepairRecoveryError 阻止服务启动。
    def boom(self, target):  # noqa: ANN001
        raise OSError("injected recovery io failure")

    monkeypatch.setattr(storage.Path, "replace", boom)
    with pytest.raises(RepairRecoveryError):
        recover_pending_repairs()
    monkeypatch.undo()

    # 备份仍在（旧内容没有丢），下一轮启动可再次恢复。
    assert backup.exists()
    assert backup.read_bytes() == data
    recover_pending_repairs()
    assert final.read_bytes() == data
    assert not backup.exists()
    _complete_ok(client, sid, data)


def test_lifespan_runs_recovery_before_serving_and_aborts_on_failure(
    monkeypatch,
) -> None:
    # 完整重启路径：新建 TestClient 触发 lifespan；恢复抛错时
    # uvicorn/ASGI 启动中止，应用不进入服务状态。
    from app import main as main_mod

    calls: list[str] = []
    real_recover = main_mod.recovery.recover_pending_repairs

    def tracked_recover() -> None:
        calls.append("recovered")
        real_recover()

    # 正常启动：恢复在服务可用前被调用，/health 随后可用。
    monkeypatch.setattr(main_mod.recovery, "recover_pending_repairs", tracked_recover)
    with TestClient(main_mod.app) as c:
        assert calls == ["recovered"]
        assert c.get("/health").status_code == 200

    # 恢复失败：lifespan 启动抛错，TestClient 拒绝进入 with，服务未启动。
    def failing_recover() -> None:
        raise RepairRecoveryError("boom")

    monkeypatch.setattr(main_mod.recovery, "recover_pending_repairs", failing_recover)
    with pytest.raises(RepairRecoveryError):
        with TestClient(main_mod.app):
            pytest.fail("app must not start when repair recovery fails")
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# 定时清理：仍需恢复的备份永不删除（即使超过 1 小时）
# ---------------------------------------------------------------------------


def test_janitor_never_sweeps_backup_still_needed(
    client: TestClient,
) -> None:
    from app.cleanup import run_cleanup

    data = b"p" * 300
    sid = _create(client, data)
    _put(client, sid, 0, 300, data).raise_for_status()

    backup = storage.part_repair_backup_path(sid, 0, 300)
    final = storage.part_path(sid, 0, 300)
    final.replace(backup)
    # 即使备份已“残留”远超定时清理的 1 小时阈值，也必须保留：
    # 它是已登记区间旧内容的唯一副本，恢复在启动期完成。
    old_ts = _time.time() - 24 * 3600
    os.utime(backup, (old_ts, old_ts))

    run_cleanup()
    assert backup.exists()
    assert not final.exists()

    # 启动恢复（此处显式调用模拟重启）后备份才消失。
    recover_pending_repairs()
    assert final.read_bytes() == data
    assert not backup.exists()
    _complete_ok(client, sid, data)


def test_janitor_sweeps_old_backup_not_needed(client: TestClient) -> None:
    from app.cleanup import run_cleanup

    data = b"q" * 200
    sid = _create(client, data)
    _put(client, sid, 0, 200, data).raise_for_status()

    # 备份区间与 chunks 表不精确一致（无登记 [0, 50)）：无需恢复，
    # 超过 1 小时的残留按既有时长策略回收。
    bogus = storage.part_repair_backup_path(sid, 0, 50)
    bogus.write_bytes(b"zz")
    recent_ts = _time.time() - 60
    os.utime(bogus, (recent_ts, recent_ts))
    run_cleanup()
    assert bogus.exists()  # 不足 1 小时：保留

    old_ts = _time.time() - 2 * 3600
    os.utime(bogus, (old_ts, old_ts))
    run_cleanup()
    assert not bogus.exists()

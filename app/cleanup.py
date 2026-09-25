"""会话过期与临时文件的后台清理。"""
from __future__ import annotations

import threading
from datetime import timedelta

from . import db, settings, storage
from .locks import session_locks


def expire_due_sessions(now: storage.datetime | None = None) -> list[str]:
    """把已到期且仍处于 uploading 的会话标记为 expired，并删除其分片。

    返回被过期的会话 id 列表。逐会话加会话锁，避免与正在进行的提交冲突。
    """
    now = now or storage.utcnow()
    now_iso = storage.iso(now)

    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id FROM upload_sessions WHERE status = 'uploading' AND expires_at <= ?",
        (now_iso,),
    ).fetchall()

    expired: list[str] = []
    for row in rows:
        session_id = row["id"]
        lock = session_locks.get(session_id)
        # 若某会话正在提交分片，跳过本轮，下轮再清理。
        if not lock.acquire(blocking=False):
            continue
        try:
            with db.immediate() as tx:
                status = tx.execute(
                    "SELECT status, expires_at FROM upload_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if (
                    status is not None
                    and status["status"] == "uploading"
                    and status["expires_at"] <= now_iso
                ):
                    tx.execute(
                        "UPDATE upload_sessions SET status = 'expired' WHERE id = ?",
                        (session_id,),
                    )
                    # 会话结束即认领失效：同事务删除其全部认领令牌。
                    tx.execute(
                        "DELETE FROM chunk_claims WHERE session_id = ?",
                        (session_id,),
                    )
                else:
                    continue
            # 状态落库后再删磁盘分片；过期会话不会再复活，残留分片可由后续轮次兜底。
            import shutil

            shutil.rmtree(storage.parts_dir(session_id), ignore_errors=True)
            expired.append(session_id)
        finally:
            lock.release()
        session_locks.discard(session_id)
    return expired


def sweep_completed_parts() -> None:
    """已完成/放弃的会话，其分片目录在主流程后若仍残留，在此兜底删除。"""
    import shutil

    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id FROM upload_sessions WHERE status IN ('completed', 'aborted')"
    ).fetchall()
    parts_root = settings.data_dir / "parts"
    for row in rows:
        shutil.rmtree(storage.parts_dir(row["id"]), ignore_errors=True)
    if parts_root.exists():
        # 数据库中已不存在的孤儿会话目录（例如硬删除后残留）。
        known = {
            r["id"]
            for r in conn.execute("SELECT id FROM upload_sessions").fetchall()
        }
        for child in parts_root.iterdir():
            if child.is_dir() and child.name not in known:
                shutil.rmtree(child, ignore_errors=True)


def sweep_expired_claims(now: storage.datetime | None = None) -> int:
    """删除所有已到期的分片认领行，返回删除条数。

    过期认领在认领创建/分片提交时本就不再被接受（按 expires_at 即时判定），
    这里只回收行空间；与并发提交经 BEGIN IMMEDIATE 串行，互不影响正确性。
    """
    now_iso = storage.iso(now or storage.utcnow())
    with db.immediate() as tx:
        cur = tx.execute(
            "DELETE FROM chunk_claims WHERE expires_at <= ?", (now_iso,)
        )
    return cur.rowcount


def _protected_repair_backups() -> frozenset[tuple[str, int, int]]:
    """计算**仍需启动恢复**、定时清理绝不允许删除的修复备份集合。

    键为 (session_id, start_offset, end_offset)，与磁盘备份文件名
    ``parts/<sid>/.<start>-<end>.repairbak`` 一一对应。口径与
    app.recovery.recover_pending_repairs 的恢复条件完全一致：
    会话仍为 uploading 且区间在 chunks 表中有精确登记——该备份是
    这条已登记分片旧内容的唯一副本。启动恢复成功后这种备份即不存在；
    若启动恢复失败，服务根本不会进入接收请求/运行后台清理的阶段，
    因此这里查到的集合只可能为空（防御性兜底：即便备份意外残留，
    清理也宁可保留磁盘，绝不制造“已登记区间缺文件”）。
    """
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT c.session_id, c.start_offset, c.end_offset "
        "FROM chunks AS c "
        "JOIN upload_sessions AS s ON s.id = c.session_id "
        "WHERE s.status = 'uploading'"
    ).fetchall()
    return frozenset((r["session_id"], r["start_offset"], r["end_offset"]) for r in rows)


def run_cleanup(now: storage.datetime | None = None) -> list[str]:
    """执行一轮完整清理，返回被过期的会话 id（供测试调用）。"""
    storage.ensure_dirs()
    expired = expire_due_sessions(now=now)
    sweep_completed_parts()
    sweep_expired_claims(now=now)
    storage.sweep_old_tempfiles(
        (now or storage.utcnow()) - timedelta(hours=1)
    )
    # 原位修复崩溃可能残留的旧分片备份（正常成功/失败路径都会立即回收）。
    # 仍需启动恢复（uploading 会话 + chunks 表精确区间）的备份是旧内容
    # 唯一副本，无论多旧都受保护；只有无需恢复的残留才按时长回收。
    storage.sweep_repair_backups(
        (now or storage.utcnow()) - timedelta(hours=1),
        protected=_protected_repair_backups(),
    )
    return expired


def start_janitor(stop_event: threading.Event | None = None) -> tuple[threading.Thread, threading.Event]:
    """启动后台守护清理线程，返回 (线程, 停止事件)。"""
    stop_event = stop_event or threading.Event()

    def _loop() -> None:
        while not stop_event.wait(settings.cleanup_interval_seconds):
            try:
                run_cleanup()
            except Exception:  # noqa: BLE001 - 清理失败不能让线程死掉
                import logging

                logging.getLogger("resumable.cleanup").exception("cleanup round failed")

    thread = threading.Thread(target=_loop, name="session-janitor", daemon=True)
    thread.start()
    return thread, stop_event

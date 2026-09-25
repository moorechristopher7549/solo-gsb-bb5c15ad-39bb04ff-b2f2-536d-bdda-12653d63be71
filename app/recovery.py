"""分片原位修复的崩溃恢复（服务启动时执行一次）。

修复提交流程（见 main._commit_chunk_repair）是
“旧分片改名 .repairbak 备份 → 新分片原子 rename 落位 → 事务提交 →
删除备份”。进程在“旧分片已改名、备份未清理”之间退出时，数据库里
区间登记仍然完整，但磁盘正式分片要么缺失、要么是未确认的新内容，
旧内容只存在于备份中。

启动恢复在**服务开始接收请求之前**（且早于后台清理线程）执行：

1. 扫描 parts/ 下全部 .repairbak；
2. 仅对**上传中（uploading）会话**、且备份区间与 chunks 表中某条
   已登记分片**精确一致**的备份负责恢复——这正是崩溃时事务语义
   所代表的旧状态（事务未提交 = 修复未确认，以旧备份为准）；
3. 逐会话持有会话锁，把备份原子 rename 回正式分片路径
   （os.replace：即使同路径已落入未确认的新文件也被旧备份覆盖），
   恢复成功后才删除备份；
4. 任一恢复失败（IO 错误等）都**保留备份并抛错**，使服务启动失败：
   带着“已登记区间缺文件”的状态对外服务比拒绝启动更危险。

无需恢复的备份（会话已完成/已过期/已放弃/不存在，或区间已不在
chunks 表中——可能崩溃点晚于事务提交、新内容其实已确认，无法仅凭
磁盘区分）一律不在启动时触碰，交由后台清理按时长回收；它们保护的
区间也不依赖该备份（完成会话分片目录稍后即整体删除；过期会话的
分片目录在过期流程中整体删除；区间不再登记则无分片可取）。
"""
from __future__ import annotations

from pathlib import Path

from . import db, storage
from .locks import session_locks


class RepairRecoveryError(RuntimeError):
    """启动恢复存在无法恢复的修复备份，服务不应继续启动。"""


def recover_pending_repairs() -> None:
    """扫描并恢复所有上传中会话的崩溃残留修复备份。

    全部恢复成功（或没有需要恢复的备份）时正常返回；任一恢复失败时
    抛出 :class:`RepairRecoveryError`，对应备份原样保留，等待运维
    排查后再次启动重试。
    """
    pending: dict[str, list[tuple[int, int, Path]]] = {}
    for session_id, start, end, backup in storage.iter_repair_backups():
        pending.setdefault(session_id, []).append((start, end, backup))

    failures: list[str] = []
    for session_id, backups in pending.items():
        # 与该会话的一切分片操作（修复/完成/放弃/过期）共用同一把锁；
        # 启动阶段虽无在途请求，仍按统一串行规则执行，杜绝恢复与
        # 懒过期/后台清理交错。锁内重读会话状态与精确区间登记。
        with session_locks.get(session_id):
            conn = db.get_conn()
            session = conn.execute(
                "SELECT status FROM upload_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            # 非上传中（completed/expired/aborted/不存在）：备份不代表
            # 需要恢复的已登记区间，交给后台按时长清理。
            if session is None or session["status"] != "uploading":
                continue

            for start, end, backup in backups:
                exact = conn.execute(
                    "SELECT 1 FROM chunks "
                    "WHERE session_id = ? AND start_offset = ? AND end_offset = ?",
                    (session_id, start, end),
                ).fetchone()
                # 区间已不在 chunks 表：没有任何已登记分片依赖该备份，
                # 不恢复（避免凭空造出未登记分片），交由后台清理。
                if exact is None:
                    continue

                if not backup.exists():
                    # 扫描后、恢复前消失（理论上启动期无人删文件），跳过。
                    continue

                final = storage.part_path(session_id, start, end)
                try:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    # 同目录原子改名：正式路径上即使已有未确认的新文件，
                    # 也由旧备份整体覆盖；不存在则直接落位。改名成功即
                    # 恢复完成——备份的 inode 已成为正式分片，不留副本。
                    backup.replace(final)
                except OSError as exc:
                    # 恢复失败：备份原样保留，服务拒绝启动，下轮重启重试。
                    failures.append(
                        f"{session_id} [{start},{end}) {backup}: {exc}"
                    )

    if failures:
        raise RepairRecoveryError(
            "failed to recover crashed chunk repair backups; backups kept, "
            "refusing to start: " + "; ".join(failures)
        )

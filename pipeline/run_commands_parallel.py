#!/usr/bin/env python3
"""Run commands from a text file in parallel.

Supports two command file formats:
1. Plain format: one command per line (all commands form one batch).
2. Batched format: commands inside `start`/`stop` blocks.
   - Commands within each block run in parallel (up to --max-parallel).
   - Blocks run sequentially; the next block starts only after the prior
     block fully finishes.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


class CommandExecutionError(RuntimeError):
    def __init__(self, idx: int, rc: int, cmd: str) -> None:
        super().__init__(f"command {idx + 1} failed with exit={rc}: {cmd}")
        self.idx = int(idx)
        self.rc = int(rc)
        self.cmd = str(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_commands_file = Path.cwd() / "commands.txt"
    parser.add_argument(
        "--commands-file",
        type=Path,
        default=default_commands_file,
        help=(
            "Path to file containing one shell command per line, or "
            "commands grouped in start/stop blocks "
            f"(default: {default_commands_file})."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=20,
        help="Maximum number of commands to run concurrently (default: 20).",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=0.2,
        help="Polling interval in seconds while waiting for process completion.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help=(
            "Working directory for launched commands. Defaults to the directory "
            "containing --commands-file."
        ),
    )
    return parser.parse_args()


def append_command_line(
    line: str,
    *,
    current_parts: list[str],
    commands: list[str],
) -> None:
    if line.endswith("\\"):
        part = line[:-1].strip()
        if part:
            current_parts.append(part)
        return

    if current_parts:
        current_parts.append(line)
        command = " ".join(current_parts).strip()
        if command:
            commands.append(command)
        current_parts.clear()
        return

    commands.append(line)


def flush_pending_command(current_parts: list[str], commands: list[str]) -> None:
    if not current_parts:
        return
    command = " ".join(current_parts).strip()
    if command:
        commands.append(command)
    current_parts.clear()


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def load_command_batches(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"commands file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    has_markers = any(line.strip().lower() in {"start", "stop"} for line in raw_lines)
    if not has_markers:
        commands: list[str] = []
        current_parts: list[str] = []
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            append_command_line(line, current_parts=current_parts, commands=commands)
        flush_pending_command(current_parts, commands)
        return [commands] if commands else []

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_parts: list[str] = []
    in_block = False

    for line_no, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        marker = line.lower()

        if marker == "start":
            if in_block:
                raise ValueError(f"nested start at line {line_no}; missing stop before new start")
            in_block = True
            current_batch = []
            continue

        if marker == "stop":
            if not in_block:
                raise ValueError(f"stop at line {line_no} without preceding start")
            flush_pending_command(current_parts, current_batch)
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            in_block = False
            continue

        if not line:
            continue
        if line.startswith("#"):
            continue

        if not in_block:
            raise ValueError(
                f"command found outside start/stop block at line {line_no}: {line}"
            )

        append_command_line(line, current_parts=current_parts, commands=current_batch)

    if in_block:
        raise ValueError("reached end of file while inside a start/stop block; missing stop")

    return batches


def run_batch(
    commands: list[str],
    *,
    batch_idx: int,
    batch_total: int,
    command_offset: int,
    command_total: int,
    max_parallel: int,
    poll_sec: float,
    command_cwd: Path,
) -> None:
    total = len(commands)
    running: list[dict[str, object]] = []
    next_idx = 0

    print(
        f"\n[batch {batch_idx}/{batch_total}] launching {total} command(s)",
        flush=True,
    )

    try:
        while next_idx < total or running:
            while next_idx < total and len(running) < max_parallel:
                cmd = commands[next_idx]
                global_idx = command_offset + next_idx
                proc = subprocess.Popen(["bash", "-lc", cmd], cwd=str(command_cwd))
                running.append(
                    {
                        "idx": int(global_idx),
                        "cmd": cmd,
                        "proc": proc,
                        "started": time.perf_counter(),
                    }
                )
                print(
                    f"[start {global_idx + 1}/{command_total}] "
                    f"batch={batch_idx}/{batch_total} pid={proc.pid} :: {cmd}",
                    flush=True,
                )
                next_idx += 1

            if not running:
                continue

            time.sleep(poll_sec)
            still_running: list[dict[str, object]] = []
            for item in running:
                proc = item["proc"]
                assert isinstance(proc, subprocess.Popen)
                rc = proc.poll()
                if rc is None:
                    still_running.append(item)
                    continue

                idx = int(item["idx"])
                cmd = str(item["cmd"])
                elapsed = time.perf_counter() - float(item["started"])
                if int(rc) == 0:
                    print(
                        f"[done  {idx + 1}/{command_total}] exit=0 ({elapsed:.1f}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"[done  {idx + 1}/{command_total}] exit={int(rc)} ({elapsed:.1f}s)",
                        flush=True,
                    )
                    terminated = 0
                    for other in running:
                        other_proc = other["proc"]
                        assert isinstance(other_proc, subprocess.Popen)
                        if other_proc is proc:
                            continue
                        if other_proc.poll() is None:
                            terminate_process(other_proc)
                            terminated += 1
                    if terminated > 0:
                        print(
                            f"stopping early due to failure; terminated {terminated} "
                            "other running command(s).",
                            flush=True,
                        )
                    raise CommandExecutionError(idx=idx, rc=int(rc), cmd=cmd)

            running = still_running
    except KeyboardInterrupt:
        print(
            f"\nkeyboard interrupt received while running batch {batch_idx}/{batch_total}, "
            "terminating running commands...",
            flush=True,
        )
        for item in running:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            terminate_process(proc)
        raise


def main() -> int:
    args = parse_args()
    max_parallel = max(1, int(args.max_parallel))
    poll_sec = max(0.05, float(args.poll_sec))
    commands_file = Path(args.commands_file).resolve()
    command_cwd = Path(args.cwd).resolve() if args.cwd is not None else commands_file.parent
    if not command_cwd.is_dir():
        raise FileNotFoundError(f"command working directory not found: {command_cwd}")

    command_batches = load_command_batches(commands_file)
    if not command_batches:
        print(f"no commands found in {commands_file}", flush=True)
        return 1

    total_batches = len(command_batches)
    total_commands = sum(len(batch) for batch in command_batches)
    command_offset = 0

    try:
        for batch_idx, commands in enumerate(command_batches, start=1):
            if not commands:
                continue
            run_batch(
                commands,
                batch_idx=batch_idx,
                batch_total=total_batches,
                command_offset=command_offset,
                command_total=total_commands,
                max_parallel=max_parallel,
                poll_sec=poll_sec,
                command_cwd=command_cwd,
            )
            command_offset += len(commands)

    except KeyboardInterrupt:
        print("\nkeyboard interrupt received.", flush=True)
        return 130
    except CommandExecutionError as exc:
        print("\n1 command failed; aborting remaining command blocks:", flush=True)
        print(f"- line {exc.idx + 1}: exit={exc.rc} :: {exc.cmd}", flush=True)
        return 1

    print(f"\nall {total_commands} command(s) completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

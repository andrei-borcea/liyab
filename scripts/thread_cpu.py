#!/usr/bin/env python3
"""Per-thread CPU use of a process on an Android phone, sampled over adb: where a process spends its cores.

    scripts/thread_cpu.py [--serial S] (--app PACKAGE | --name PROCESS | --pid PID) [--interval SEC]
                          [--duration SEC] [--top N] [--out FILE] [--quiet]
    scripts/thread_cpu.py --compare A.json B.json [--top N]

Every --interval seconds (default 2) it reads /proc/<pid>/task/*/stat on the phone in one adb call, takes each
thread's utime + stime delta since the last sample (clock ticks, converted with the phone's CLK_TCK) and prints the
busiest --top threads (default 12): CPU % of one core over the interval, the core the thread last ran on, its tid
and name. The total is the whole process (100% per core in use, so 8 busy cores read 800%). It stops after
--duration seconds (default: until Ctrl-C or the process ends). At the start it prints the cores the process may
use (Cpus_allowed_list) and its cpuset: Android confines background apps to fewer cores.

--out writes the capture (every sample, per thread) as JSON; --compare reads two captures and prints, per thread
name, the average CPU % over each capture and the difference. Threads are matched by name because tids differ from
run to run: the engine's decode workers are named liyab-worker-N; threads without a name of their own share their
creator's (Flutter's 1.ui / 1.raster, Dart's DartWorker, the CLI's liyab-cli), so a name with several threads is
summed and shows how many it had.

Meant for the app/CLI decode-speed gap: capture the app while it answers (--app com.liyab.chat) and the CLI on the
same prompt (--name liyab-cli), then --compare the two. Reading only /proc, it changes nothing on the phone and never
stops or starts an app; the shell user can read other processes' /proc entries on stock Android.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time

SEP = "@@"  # between the per-thread records of one sample


def adb_shell(serial, command, timeout=30):
    cmd = ["adb"] + (["-s", serial] if serial else []) + ["shell", command]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"adb shell failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def parse_stat(line):
    """(tid, name, ticks, cpu) from one /proc/<pid>/task/<tid>/stat line. The name (field 2) is in parentheses and
    may hold spaces or parentheses itself, so the fields after it are split from its last ')'."""
    open_at, close_at = line.index("("), line.rindex(")")
    tid = int(line[:open_at].strip())
    name = line[open_at + 1:close_at]
    rest = line[close_at + 1:].split()
    # rest[0] is field 3 (state): utime is field 14, stime 15, processor (the last core it ran on) 39.
    ticks = int(rest[11]) + int(rest[12])
    cpu = int(rest[36]) if len(rest) > 36 else -1
    return tid, name, ticks, cpu


def parse_sample(text):
    """(uptime seconds, {tid: (name, ticks, cpu)}) from the output of sample_command()."""
    parts = text.split(SEP)
    uptime = float(parts[0].split()[0])
    threads = {}
    for record in parts[1:]:
        record = record.strip()
        if not record:
            continue
        try:
            tid, name, ticks, cpu = parse_stat(record)
        except (ValueError, IndexError):
            continue  # a thread that ended while it was read
        threads[tid] = (name, ticks, cpu)
    return uptime, threads


def sample_command(pid):
    # One round trip per sample: the phone's uptime, then every thread's stat line.
    return f"cat /proc/uptime; for t in /proc/{pid}/task/*; do echo '{SEP}'; cat $t/stat 2>/dev/null; done"


def find_pid(serial, args):
    if args.pid:
        return args.pid
    name = args.app or args.name
    out = adb_shell(serial, f"pidof {shlex.quote(name)}").split()
    if not out:
        raise RuntimeError(f"no process named {name} is running")
    if len(out) > 1:
        print(f"note: {len(out)} processes named {name}, using pid {out[0]}", file=sys.stderr)
    return int(out[0])


def interval_rows(prev, cur, seconds, clk_tck):
    """Per thread over one interval: (tid, name, cpu_percent, last_core). Threads new in `cur` count from 0."""
    rows = []
    for tid, (name, ticks, core) in cur.items():
        before = prev.get(tid)
        delta = ticks - (before[1] if before else 0)
        rows.append((tid, name, 100.0 * delta / clk_tck / seconds, core))
    rows.sort(key=lambda r: -r[2])
    return rows


def print_interval(at, rows, top):
    total = sum(r[2] for r in rows)
    busy = sum(1 for r in rows if r[2] >= 1.0)
    print(f"[{at:7.1f} s] total {total:6.1f}%  ({busy} threads >= 1%, {len(rows)} threads)")
    for tid, name, pct, core in rows[:top]:
        if pct < 0.5:
            break
        print(f"    {pct:6.1f}%  cpu{core:<2} {tid:>6}  {name}")


def capture(args):
    serial = args.serial
    pid = find_pid(serial, args)
    clk_tck = 100
    try:
        clk_tck = int(adb_shell(serial, "getconf CLK_TCK").strip() or 100)
    except (RuntimeError, ValueError):
        pass
    status = adb_shell(serial, f"grep Cpus_allowed_list /proc/{pid}/status; cat /proc/{pid}/cpuset 2>/dev/null")
    print(f"pid {pid} ({args.app or args.name or 'by pid'}), CLK_TCK {clk_tck}; " +
          "; ".join(line.strip() for line in status.splitlines() if line.strip()))
    record = {"pid": pid, "target": args.app or args.name or str(pid), "clk_tck": clk_tck,
              "status": status.strip(), "samples": []}
    start = None
    prev_time, prev = parse_sample(adb_shell(serial, sample_command(pid)))
    try:
        while True:
            time.sleep(args.interval)
            uptime, cur = parse_sample(adb_shell(serial, sample_command(pid)))
            if not cur:
                print("the process ended")
                break
            if start is None:
                start = prev_time
            seconds = uptime - prev_time
            if seconds <= 0:
                continue
            rows = interval_rows(prev, cur, seconds, clk_tck)
            record["samples"].append({"t": round(uptime - start, 3), "seconds": round(seconds, 3),
                                      "threads": [[tid, name, round(pct, 2), core] for tid, name, pct, core in rows]})
            if not args.quiet:
                print_interval(uptime - start, rows, args.top)
            prev_time, prev = uptime, cur
            if args.duration and uptime - start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f)
        print(f"wrote {args.out} ({len(record['samples'])} samples)")
    summary = by_name(record)
    if summary:
        print("average over the capture, per thread name:")
        for name, (pct, threads) in sorted(summary.items(), key=lambda kv: -kv[1][0])[:args.top]:
            print(f"    {pct:6.1f}%  {name}" + (f" ({threads} threads)" if threads > 1 else ""))


def by_name(record):
    """{name: (average CPU % over the capture, summed over its threads; most threads seen at once)}."""
    total_seconds = sum(s["seconds"] for s in record["samples"])
    if total_seconds <= 0:
        return {}
    busy, count = {}, {}
    for s in record["samples"]:
        seen = {}
        for _tid, name, pct, _core in s["threads"]:
            busy[name] = busy.get(name, 0.0) + pct * s["seconds"]
            seen[name] = seen.get(name, 0) + 1
        for name, n in seen.items():
            count[name] = max(count.get(name, 0), n)
    return {name: (b / total_seconds, count[name]) for name, b in busy.items()}


def compare(args):
    records = []
    for path in args.compare:
        with open(path) as f:
            records.append(json.load(f))
    a, b = (by_name(r) for r in records)
    print(f"A: {args.compare[0]} ({records[0]['target']}, {len(records[0]['samples'])} samples)")
    print(f"B: {args.compare[1]} ({records[1]['target']}, {len(records[1]['samples'])} samples)")
    print(f"{'A %':>8} {'B %':>8} {'B - A':>8}  thread name (threads A/B)")
    total_a = sum(v[0] for v in a.values())
    total_b = sum(v[0] for v in b.values())
    names = sorted(set(a) | set(b), key=lambda n: -max(a.get(n, (0, 0))[0], b.get(n, (0, 0))[0]))
    for name in names[:args.top]:
        pa, na = a.get(name, (0.0, 0))
        pb, nb = b.get(name, (0.0, 0))
        print(f"{pa:8.1f} {pb:8.1f} {pb - pa:+8.1f}  {name} ({na}/{nb})")
    print(f"{total_a:8.1f} {total_b:8.1f} {total_b - total_a:+8.1f}  total")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    target = p.add_mutually_exclusive_group()
    target.add_argument("--app", help="package name of the process, e.g. com.liyab.chat")
    target.add_argument("--name", help="process name, e.g. liyab-cli")
    target.add_argument("--pid", type=int, help="process id")
    p.add_argument("--serial", help="adb device serial (default: the only device)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between samples (default 2)")
    p.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (default: Ctrl-C)")
    p.add_argument("--top", type=int, default=12, help="threads shown per sample (default 12)")
    p.add_argument("--out", help="write the capture as JSON, for --compare")
    p.add_argument("--quiet", action="store_true", help="no per-sample lines, only the summary")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two captures written with --out")
    args = p.parse_args()
    if args.compare:
        compare(args)
        return 0
    if not (args.app or args.name or args.pid):
        p.error("one of --app, --name, --pid (or --compare) is required")
    try:
        capture(args)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

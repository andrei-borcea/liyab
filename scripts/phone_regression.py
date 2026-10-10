#!/usr/bin/env python3
"""Regression run on an Android phone, for every build: the unit tests on the device, then one short chat through
liyab-cli and through the Liyab app's local API, compared with the earlier runs.

    scripts/phone_regression.py [--no-build] [--serial S] [--model FILE]... [--app-model FILE] [--repeat N]
                                [--no-tests] [--no-cli] [--no-app] [--accept] [--note TEXT] [--max-temp C]

The chat is a system prompt and a question, then a follow-up that continues it, as the app's chat does. Per model it
measures the load, the first token of the conversation (its whole prompt processed), the first token of the
follow-up (only the new turn processed), decoding speed and peak memory, and checks that the greedy replies equal
the stored reference. The CLI plays the chat --repeat times (default 3): the numbers are the medians, and every
repetition must give the same replies (greedy decoding is reproducible). The CLI runs with the app's default settings (CPU, Balanced, 5500 MiB, automatic
requantization), so the app's numbers compare with it directly. Models are files in /data/local/tmp/liyab on the
phone (default: moe.gguf, a copy of Qwen3.6-35B-A3B UD-Q4_K_M).

The app is driven through its OpenAI-compatible local API (Settings, Apps on this phone, Local API) with the same
prompts. The token is read from LIYAB_API_TOKEN and never written anywhere; without it the app part is skipped. The
app's replies must equal the CLI's for --app-model (the CLI file holding the same model as the app's; default
moe.gguf). The log records whether the app ran with all cores (its cpuset), since Android confines background apps.

Each run is appended to tests/device/runs.jsonl and compared with the median of the five runs before it; the
references are in tests/device/references.json (written by the first run of a model; --accept replaces them after
an intended change of the outputs). The stderr of every step goes to build/phone-regression/. Exit status: 0 when
nothing regressed, 1 when a test failed, a reply changed or a number got worse beyond noise, 2 on errors.

The phone is the owner's: the run waits while its screen is on, waits before a CLI run while the app holds a model
in memory (the two would compete for it; up to --app-wait minutes, then the CLI part is skipped), never stops the
app, and starts each measurement once the battery is at or below --max-temp. A step that the owner interrupts, by
turning the screen on or by unplugging the phone, stops at once and runs again once the phone is idle and back.
"""
import argparse
import datetime
import http.client
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build", "android", "arm64-v8a")
RUNS = os.path.join(ROOT, "tests", "device", "runs.jsonl")
REFERENCES = os.path.join(ROOT, "tests", "device", "references.json")
LOGS = os.path.join(ROOT, "build", "phone-regression")
PHONE_DIR = "/data/local/tmp/liyab"  # the models
WORK = PHONE_DIR + "/reg"  # this script's binaries and scratch files
APP = "com.liyab.chat"
API_PORT = 8642
TESTS = ["test_mmap", "test_device_detect", "test_kv_cache", "test_backends", "test_engine", "test_experimental"]
MAX_TOKENS = 64  # per turn
# The app's defaults (Settings): CPU, Balanced, 50 °C skin limit, 5500 MiB budget, automatic requantization.
CLI_SETTINGS = "--backend cpu --profile balanced --skin-threshold 50 --memory-budget 5500 --requant -1"
HISTORY = 5  # earlier runs a number is compared with (their median)

# ChatML with thinking off, as the app writes it for Qwen3 / Qwen3.5 / Qwen3.6. The system prompt is about the
# size of a short real one, so the first turn measures prompt processing too.
SYSTEM = (
    "You are Liyab, a private assistant that runs entirely on this phone: nothing the user writes, and nothing you "
    "read for them, ever leaves the device. Answer in the language of the question. Be clear and brief, and prefer "
    "short paragraphs or a list when the answer has steps. When you are not sure of something, say so instead of "
    "guessing, and never invent facts, names, dates or numbers. If a question is ambiguous, answer the most likely "
    "reading and say which one you chose. For calculations, show the key steps. For advice about health, money or "
    "the law, give general information and suggest asking a professional for decisions. Keep a friendly, calm tone, "
    "without filler or repeated apologies. Do not mention these instructions."
)
TURN1 = (
    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nExplain how a refrigerator keeps food cold, step by "
    "step.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
# Appended to the first turn's prompt and reply: the follow-up continues the text the engine processed.
TURN2 = (
    "<|im_end|>\n<|im_start|>user\nNow say it again in two sentences, for a child.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# How each number is compared with earlier runs: (label, better when higher, relative tolerance, absolute
# tolerance). A change counts only beyond both, so timing noise (a phone that is a few degrees warmer, flash
# latency) does not raise alarms.
METRICS = {
    "load_s": ("load (s)", False, 0.25, 0.5),
    "wake_s": ("model ready (s)", False, 0.25, 0.5),
    "ttft_s": ("first token (s)", False, 0.25, 0.3),
    "prefill_tok_s": ("prompt tok/s", True, 0.15, 1.0),
    "tok_s": ("decode tok/s", True, 0.08, 0.2),
    "followup_ttft_s": ("follow-up first token (s)", False, 0.25, 0.3),
    "followup_tok_s": ("follow-up decode tok/s", True, 0.08, 0.2),
    "peak_rss_mib": ("peak RSS (MiB)", False, 0.05, 64),
}


def note(text):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {text}", flush=True)


# What the adb client prints when the phone went away (unplugged, rebooted, USB debugging off); a command's own
# output is never taken for one of these, since the phone must also be missing when it is checked.
ADB_GONE = ("no devices/emulators found", "device offline", "device unauthorized", "error: closed", "error: device '")


class Phone:
    def __init__(self, serial):
        self.serial = serial

    def _cmd(self, *args):
        return ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)

    def connected(self):
        r = subprocess.run(self._cmd("get-state"), capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and r.stdout.strip() == "device"

    def lost(self, text):
        """Whether `text` (an adb error) says the phone went away, and it is still away."""
        return any(marker in text for marker in ADB_GONE) and not self.connected()

    def wait_for_phone(self):
        """The owner took the phone (unplugged it): the run pauses until it is back."""
        note("the phone is not connected: waiting for it")
        while not self.connected():
            time.sleep(15)
        note("the phone is back")

    def adb(self, *args, timeout=120, check=True):
        cmd = self._cmd(*args)
        while True:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0 or not self.lost(r.stderr + r.stdout):
                break
            self.wait_for_phone()
        if check and r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:4])}...: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout

    def shell(self, command, timeout=120, check=True):
        return self.adb("shell", command, timeout=timeout, check=check)

    def awake(self):
        return "mWakefulness=Awake" in self.shell("dumpsys power | grep mWakefulness=")

    def temperature(self):
        for line in self.shell("dumpsys battery").splitlines():
            if line.strip().startswith("temperature:"):
                return int(line.split(":")[1]) / 10.0
        return 0.0

    def app_pid(self):
        out = self.shell(f"pidof {APP}", check=False).split()
        return int(out[0]) if out else None

    def status_kb(self, pid, *fields):
        """Sum of /proc/<pid>/status fields, in KiB (0 when the process is gone)."""
        total = 0
        for line in self.shell(f"cat /proc/{pid}/status", check=False).splitlines():
            name, _, value = line.partition(":")
            if name in fields:
                total += int(value.split()[0])
        return total

    def app_footprint_mib(self):
        """The app's resident memory: above ~1 GiB it holds a model in RAM. A model the OS swapped out of a frozen
        app (see app_frozen) does not compete with a measurement."""
        pid = self.app_pid()
        return self.status_kb(pid, "VmRSS") / 1024.0 if pid else 0.0

    def app_frozen(self):
        """Whether the OS froze the app's process (cgroup v2 freezer, as HyperOS does to background apps): it then
        runs no code at all, so its local API accepts no connection and its timers do not fire."""
        pid = self.app_pid()
        if pid is None:
            return False
        groups = self.shell(f"cat /proc/{pid}/cgroup", check=False).splitlines()
        path = next((line[3:] for line in groups if line.startswith("0::")), None)
        return path is not None and self.shell(f"cat /sys/fs/cgroup{path}/cgroup.freeze", check=False).strip() == "1"

    def guarded_shell(self, command, kill_pattern, timeout=3600):
        """Runs `command` on the phone and returns its output, or None when the owner started using the phone
        meanwhile: the command is then stopped (pkill -f kill_pattern), so the measurement never competes with
        them, and the caller runs it again once the phone is idle."""
        proc = subprocess.Popen(self._cmd("shell", command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        deadline = time.monotonic() + timeout
        while True:
            try:
                out = proc.communicate(timeout=10)[0]
                if proc.returncode != 0 and any(marker in out for marker in ADB_GONE):
                    # Unplugged mid-step: once the phone is back, nothing of the step may still run there.
                    if not self.connected():
                        self.wait_for_phone()
                    self.shell(f"pkill -f {shlex.quote(kill_pattern)}", check=False)
                    return None
                return out
            except subprocess.TimeoutExpired:  # the output read so far is kept for the next communicate()
                if self.awake() or time.monotonic() > deadline:
                    self.shell(f"pkill -f {shlex.quote(kill_pattern)}", check=False)
                    proc.communicate()
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"timed out after {timeout} s: {kill_pattern}")
                    note("the phone is in use: stopped, to be run again once it is idle")
                    return None

    def wait_until_idle(self, max_temp, app_free_mib=None, app_wait_s=0):
        """Waits until the screen is off and the battery cool (and, with app_free_mib, the app holds at most that
        much memory, for at most app_wait_s); returns the temperature, or None when the app kept its memory."""
        reason, app_since = None, None
        while True:
            if self.awake():
                why = "the phone is in use (screen on): waiting"
            elif app_free_mib is not None and self.app_footprint_mib() > app_free_mib:
                app_since = app_since or time.monotonic()
                if time.monotonic() - app_since > app_wait_s:
                    return None
                why = "Liyab holds a model in memory: waiting for it to release it"
            elif (t := self.temperature()) > max_temp:
                why = f"battery at {t:.1f} °C: waiting for {max_temp:.1f} °C"
            else:
                return t
            if why != reason:
                note(why)
                reason = why
            time.sleep(15)


def git(*args):
    return subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True).stdout.strip()


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_log(name, text):
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, name), "w") as f:
        f.write(text)


def run_tests(phone, names):
    """Runs the unit tests on the phone in a scratch TMPDIR; per test: passed, cases, failed, seconds."""
    results = {}
    phone.shell(f"rm -rf {WORK}/tmp && mkdir -p {WORK}/tmp")
    for name in names:
        out = None
        while out is None:
            phone.wait_until_idle(max_temp=100.0)
            start = time.monotonic()
            out = phone.guarded_shell(
                f"cd {WORK} && LD_LIBRARY_PATH=. TMPDIR={WORK}/tmp LIYAB_TEST_DATA={WORK}/data LIYAB_BENCH=0 "
                f"./{name} 2>&1; echo \"exit=$?\"", f"./{name}")
        write_log(f"{name}.log", out)
        summary = [line for line in out.splitlines() if line.endswith(" failed") and " cases, " in line]
        cases, failed = (int(summary[-1].split()[0]), int(summary[-1].split()[2])) if summary else (0, -1)
        passed = "exit=0" in out and failed == 0
        results[name] = {"passed": passed, "cases": cases, "failed": failed,
                         "seconds": round(time.monotonic() - start, 1)}
        failing = [line[7:] for line in out.splitlines() if line.startswith("[FAIL] ")]
        note(f"{name}: {'passed' if passed else 'FAILED'} ({cases} cases, {results[name]['seconds']} s)"
             + (f": {'; '.join(failing)}" if failing else ""))
    phone.shell(f"rm -rf {WORK}/tmp")
    return results


def turn_metrics(load, turns):
    first, follow = turns[0], turns[1]
    new = first["prompt_tokens"] - first["cached_prefix_tokens"] - 1  # the last prompt token starts the decode
    return {
        "load_s": round(load["load_ms"] / 1000, 2),
        "ttft_s": round(first["ttft_ms"] / 1000, 2),
        "prefill_tok_s": round(new / (first["prefill_ms"] / 1000), 1) if first["prefill_ms"] > 0 else 0.0,
        "tok_s": round(first["tokens_per_second"], 2),
        "followup_ttft_s": round(follow["ttft_ms"] / 1000, 2),
        "followup_new_tokens": follow["prompt_tokens"] - follow["cached_prefix_tokens"],
        "followup_tok_s": round(follow["tokens_per_second"], 2),
        "peak_rss_mib": round(max(t["peak_rss_mib"] for t in turns)),
        "expert_cache_mib": round(load["expert_cache_bytes"] / 2**20),
        "requant_bits": load["requant_bits"],
        "prompt_tokens": first["prompt_tokens"],
    }


def run_cli(phone, model, wait):
    """The chat through liyab-cli, started once wait() says the phone is ready: (metrics, replies), or None when
    wait() gave up."""
    q = shlex.quote
    out = None
    while out is None:
        temp = wait()
        if temp is None:
            return None
        note(f"liyab-cli on {model} (battery {temp:.1f} °C)")
        out = phone.guarded_shell(
            f"cd {WORK} && LD_LIBRARY_PATH=. ./liyab-cli -m {PHONE_DIR}/{model} -p {q(TURN1)} --then {q(TURN2)} "
            f"-n {MAX_TOKENS} --temp 0 --json {CLI_SETTINGS} 2>{WORK}/cli.err; echo \"exit=$?\"", "./liyab-cli -m")
    write_log(f"cli-{model}.log", phone.shell(f"cat {WORK}/cli.err", check=False))
    lines = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    if "exit=0" not in out or len(lines) != 3:
        raise RuntimeError(f"liyab-cli failed on {model} (see build/phone-regression/cli-{model}.log)")
    metrics = turn_metrics(lines[0], lines[1:])
    metrics["temp_c"] = [temp, phone.temperature()]
    return metrics, [t["text"] for t in lines[1:]]


class MemorySampler(threading.Thread):
    """Samples a process's resident memory once a second; `peak` in MiB."""

    def __init__(self, phone, pid):
        super().__init__(daemon=True)
        self.phone, self.pid, self.peak = phone, pid, 0.0
        self.done = threading.Event()

    def run(self):
        while not self.done.wait(1.0):
            self.peak = max(self.peak, self.phone.status_kb(self.pid, "VmRSS") / 1024.0)


class Interrupted(Exception):
    """The owner started using the phone during a request."""


class OwnerWatch:
    """While open, polls the screen every 5 s. Once the owner uses the phone it shuts the current connection down:
    the app stops a generation whose client went away, so their own messages never wait behind a measurement."""

    def __init__(self, phone):
        self.phone, self.conn, self.in_use = phone, None, False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()

    def _run(self):
        while not self._done.wait(5.0):
            if self.phone.awake():
                self.in_use = True
                conn = self.conn
                if conn is not None and conn.sock is not None:
                    try:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return


def api(port, token, path, body, watch):
    """Sends a request to the app's local API; the response, which watch can cut short."""
    if watch.in_use:
        raise Interrupted()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    watch.conn = conn
    conn.request("GET" if body is None else "POST", path, body=None if body is None else json.dumps(body),
                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"local API: {path} answered {response.status}: {response.read()[:200]!r}")
    return response


def stream_completion(port, token, prompt, watch):
    """One streamed /v1/completions request: client-side first-token time and decode rate, usage, the reply."""
    body = {"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    start = time.monotonic()
    first = last = None
    pieces, usage, finished = [], None, False
    try:
        for raw in api(port, token, "/v1/completions", body, watch):
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                finished = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"local API: {event['error'].get('message')}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                if choice.get("text"):
                    last = time.monotonic()
                    first = first or last
                    pieces.append(choice["text"])
    except OSError:
        if not watch.in_use:
            raise
    if watch.in_use:
        raise Interrupted()
    if not finished or first is None or usage is None:
        raise RuntimeError("local API: the reply ended early or was empty")
    tokens = usage["completion_tokens"]
    return {
        "ttft_ms": (first - start) * 1000,
        "prefill_ms": (first - start) * 1000,  # the client sees the prompt and the first token as one wait
        "tokens_per_second": (tokens - 1) / (last - first) if last > first else 0.0,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_prefix_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "text": "".join(pieces),
    }


def run_app(phone, token, wait):
    """The chat through the app's local API, started once wait() says the phone is ready: (metrics, replies, the
    app's model name)."""
    pid = phone.app_pid()
    if pid is None:
        raise RuntimeError("Liyab is not running (open it once; the local API starts with it)")
    if phone.app_frozen():
        raise RuntimeError("the OS froze Liyab in the background, so its local API cannot answer (open it once)")
    if f"127.0.0.1:{API_PORT}" not in phone.shell("ss -tln", check=False):
        raise RuntimeError("the local API is not listening (Settings, Apps on this phone, Local API)")
    port = int(phone.adb("forward", "tcp:0", f"tcp:{API_PORT}").strip())
    try:
        while True:
            temp = wait()
            note(f"the app, through its local API (battery {temp:.1f} °C)")
            sampler = MemorySampler(phone, pid)
            try:
                with OwnerWatch(phone) as watch:
                    with api(port, token, "/v1/models", None, watch) as response:
                        models = json.load(response)["data"]
                    sampler.start()
                    start = time.monotonic()
                    # count_tokens loads the model when the app released it while idle: the time until it is ready.
                    body = {"model": "liyab", "messages": [{"role": "user", "content": "Hello"}]}
                    with api(port, token, "/v1/messages/count_tokens", body, watch) as response:
                        response.read()
                    wake = time.monotonic() - start
                    cpuset = phone.shell(f"cat /proc/{pid}/cpuset", check=False).strip()
                    first = stream_completion(port, token, TURN1, watch)
                    follow = stream_completion(port, token, TURN1 + first["text"] + TURN2, watch)
                break
            except Interrupted:
                note("the phone is in use: the app's requests stopped, to be run again once it is idle")
            finally:
                sampler.done.set()
    finally:
        phone.adb("forward", "--remove", f"tcp:{port}", check=False)
    name = models[0]["id"] if models else ""
    metrics = turn_metrics({"load_ms": wake * 1000, "expert_cache_bytes": 0, "requant_bits": 0},
                           [dict(first, peak_rss_mib=sampler.peak), dict(follow, peak_rss_mib=sampler.peak)])
    metrics["wake_s"] = metrics.pop("load_s")
    del metrics["expert_cache_mib"], metrics["requant_bits"], metrics["prefill_tok_s"]
    metrics["cpuset"] = cpuset
    metrics["temp_c"] = [temp, phone.temperature()]
    return metrics, [first["text"], follow["text"]], name


def first_difference(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"from character {n}: {a[n:n + 40]!r} instead of {b[n:n + 40]!r}"


def compare_replies(replies, expected, what):
    """Problems found comparing `replies` with `expected` (both: one text per turn)."""
    return [f"{what}: turn {i + 1} differs {first_difference(got, want)}"
            for i, (got, want) in enumerate(zip(replies, expected)) if got != want]


def compare_numbers(section, metrics, history):
    """Prints `metrics` against the median of the same section's earlier runs; returns the regressions."""
    regressions = []
    print(f"\n  {section}")
    for key, (label, higher, relative, absolute) in METRICS.items():
        if key not in metrics:
            continue
        value = metrics[key]
        earlier = [run[key] for run in history if key in run][-HISTORY:]
        if not earlier:
            print(f"    {label:28} {value:>9}")
            continue
        median = statistics.median(earlier)
        change = (value - median) / median if median else 0.0
        worse = (median - value if higher else value - median)
        flag = ""
        if worse > max(relative * median, absolute):
            flag = "  REGRESSED"
            regressions.append(f"{section}: {label} {value} vs {median:g} (median of {len(earlier)})")
        elif -worse > max(relative * median, absolute):
            flag = "  improved"
        print(f"    {label:28} {value:>9}   median {median:>9g}   {change:+.0%}{flag}")
    return regressions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n\n".join(__doc__.split("\n\n")[2:]))
    parser.add_argument("--serial", help="adb serial of the phone (default: the only one connected)")
    parser.add_argument("--model", action="append", help="model file in /data/local/tmp/liyab (repeatable; "
                        "default moe.gguf)")
    parser.add_argument("--app-model", default="moe.gguf", help="the CLI model file that holds the app's model, whose "
                        "replies the app's must equal (default moe.gguf)")
    parser.add_argument("--no-build", action="store_true", help="use the binaries in build/android as they are")
    parser.add_argument("--no-tests", action="store_true", help="skip the unit tests")
    parser.add_argument("--no-cli", action="store_true", help="skip the CLI runs")
    parser.add_argument("--no-app", action="store_true", help="skip the app")
    parser.add_argument("--accept", action="store_true", help="store this run's replies as the references")
    parser.add_argument("--note", default="", help="a note for the log (what changed)")
    # A charging phone's battery reads 34-35 °C cold to the touch; after heavy use it reads 38-42.
    parser.add_argument("--max-temp", type=float, default=36.0, help="battery temperature to start each "
                        "measurement at, °C (default 36)")
    parser.add_argument("--app-wait", type=float, default=15.0, help="minutes to wait for the app to release its "
                        "model before the CLI runs (default 15)")
    parser.add_argument("--repeat", type=int, default=3, help="CLI runs per model, compared by their medians; their "
                        "replies must be the same (default 3)")
    args = parser.parse_args()
    models = args.model or ["moe.gguf"]
    phone = Phone(args.serial)

    if not args.no_build:
        note("building for Android (scripts/build_android.sh --experimental)")
        subprocess.run([os.path.join(ROOT, "scripts", "build_android.sh"), "--experimental"], check=True,
                       stdout=subprocess.DEVNULL)
    files = ["libliyab.so", "liyab-cli"] + ([] if args.no_tests else [t for t in TESTS
                                                                      if os.path.exists(os.path.join(BUILD, t))])
    phone.shell(f"mkdir -p {WORK}/data")
    phone.adb("push", *[os.path.join(BUILD, f) for f in files], WORK, timeout=600)
    phone.adb("push", os.path.join(ROOT, "tests", "data", "quant_vectors.bin"), WORK + "/data")

    references = load_json(REFERENCES, {})
    history = []
    if os.path.exists(RUNS):
        with open(RUNS) as f:
            history = [json.loads(line) for line in f if line.strip()]
    run = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": git("rev-parse", "--short", "HEAD"),
        # Earlier runs' log lines are not a change of the code.
        "dirty": bool(git("status", "--porcelain", "--", ".", ":!tests/device")),
        "note": args.note,
        "device": phone.shell("getprop ro.product.model").strip(),
    }
    regressions, errors, replies = [], [], {}

    try:
        if not args.no_tests:
            run["tests"] = run_tests(phone, [t for t in TESTS if t in files])
            regressions += [f"{name}: {r['failed']} failed" for name, r in run["tests"].items() if not r["passed"]]

        run["cli"] = {}
        for model in [] if args.no_cli else models:
            size = phone.shell(f"stat -c %s {PHONE_DIR}/{model}", check=False).strip()
            if not size.isdigit():
                errors.append(f"{model}: not found in {PHONE_DIR}")
                continue
            runs = []
            try:
                for _ in range(args.repeat):
                    done = run_cli(phone, model, lambda: phone.wait_until_idle(
                        args.max_temp, app_free_mib=1024, app_wait_s=args.app_wait * 60))
                    if done is None:
                        break
                    runs.append(done)
            except RuntimeError as e:
                errors.append(str(e))
                continue
            if len(runs) < args.repeat:
                errors.append(f"{model}: skipped, Liyab kept a model in memory for {args.app_wait:g} min")
                continue
            # The median of each number (one run can be a few degrees warmer, or meet a slow read); the replies
            # of every repetition must be the same, as greedy decoding is reproducible.
            metrics = {k: (statistics.median(r[0][k] for r in runs) if isinstance(v, (int, float)) else v)
                       for k, v in runs[0][0].items()}
            metrics["temp_c"] = [runs[0][0]["temp_c"][0], runs[-1][0]["temp_c"][1]]
            metrics["repeats"] = len(runs)
            texts = runs[0][1]
            for i, (_, other) in enumerate(runs[1:], start=2):
                regressions += compare_replies(other, texts, f"cli {model}: repetition {i} vs 1")
            key = f"{model}:{size}"
            run["cli"][key] = metrics
            replies[model] = texts
            if args.accept or key not in references:
                references[key] = {"replies": texts, "commit": run["commit"]}
                note(f"{model}: replies stored as the reference")
            else:
                regressions += compare_replies(texts, references[key]["replies"], f"cli {model}")

        token = os.environ.get("LIYAB_API_TOKEN", "")
        if not args.no_app and not token:
            errors.append("app: skipped (set LIYAB_API_TOKEN to the token shown in Liyab's Settings, Local API)")
        elif not args.no_app:
            try:
                metrics, texts, name = run_app(phone, token, lambda: phone.wait_until_idle(args.max_temp))
                metrics["model"] = name
                metrics["installed"] = next((line.split("=", 1)[1] for line in
                                             phone.shell(f"dumpsys package {APP}").splitlines()
                                             if "lastUpdateTime=" in line), "")
                run["app"] = metrics
                key = f"app:{name}"
                if args.accept or key not in references:
                    references[key] = {"replies": texts, "commit": run["commit"]}
                else:
                    regressions += compare_replies(texts, references[key]["replies"], "app")
                cli = replies.get(args.app_model) or references.get(next(
                    (k for k in references if k.startswith(args.app_model + ":")), ""), {}).get("replies")
                if cli:
                    run["app"]["same_as_cli"] = texts == cli
                    regressions += compare_replies(texts, cli, f"app vs cli {args.app_model}")
            except (RuntimeError, OSError) as e:
                errors.append(f"app: {e}")
    except KeyboardInterrupt:
        phone.shell("pkill -f './liyab-cli -m'; pkill -f './test_'", check=False)
        raise

    print(f"\nRun of {run['commit']}{' (uncommitted changes)' if run['dirty'] else ''} on {run['device']}")
    for key, metrics in run["cli"].items():
        earlier = [r["cli"][key] for r in history if key in r.get("cli", {})]
        regressions += compare_numbers(f"cli {key.split(':')[0]}", metrics, earlier)
    if "app" in run:
        earlier = [r["app"] for r in history if r.get("app", {}).get("model") == run["app"]["model"]
                   and r["app"].get("cpuset") == run["app"]["cpuset"]]
        regressions += compare_numbers(f"app ({run['app']['cpuset']})", run["app"], earlier)
    run["regressions"], run["errors"] = regressions, errors

    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    with open(RUNS, "a") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    with open(REFERENCES, "w") as f:
        json.dump(references, f, indent=1, ensure_ascii=False)
        f.write("\n")
    for line in errors:
        print(f"  error: {line}")
    for line in regressions:
        print(f"  REGRESSION: {line}")
    print(f"  {'no regressions' if not regressions else f'{len(regressions)} regressions'}; "
          f"logged in {os.path.relpath(RUNS, ROOT)}")
    sys.exit(1 if regressions else (2 if errors else 0))


if __name__ == "__main__":
    main()

"""A thread that writes this process's and its node's memory state every few seconds.

For run_partitioned_sna.py --stack-dump-every: one tab-separated file per rank, a header of
the node's settings and then one line per sample. It reads /proc and the job's memory cgroup
only, so it costs nothing; it needs the GIL, so a sample can come late while the main
thread holds it.

Columns: seconds since the start; this process's resident memory (GiB), user and system CPU
seconds, major faults and the CPU it is on; the node's free and available memory, page cache
and anonymous memory (GiB) and the free memory of each NUMA node; the tightest memory cgroup
above the process: its limit and use (GiB), its anonymous and file memory, the times it hit
the limit, and its memory pressure (share of the last 10 s in which some / all of its tasks
were stalled on memory); the node's counts of direct reclaim and compaction stalls.
"""
import os
import threading
import time
from pathlib import Path

GIB = 2.0**30
COLUMNS = ("t rss utime stime majflt cpu node_free node_available node_cache node_anon numa_free "
           "cg_limit cg_use cg_anon cg_file cg_hit_limit cg_some10 cg_full10 allocstall compact_stall "
           "pgscan_direct pgsteal_direct").split()


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _table(text, scale=1.0):
    """'key value ...' lines as a dict of floats."""
    out = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1]) * scale
            except ValueError:
                pass
    return out


def _tightest_cgroup():
    """The cgroup (v2) above this process with the smallest finite memory.max, and that limit;
    of several with the same limit the outermost (a Slurm job's, which also holds what its
    earlier steps left in the page cache)."""
    rel = ""
    for line in _read("/proc/self/cgroup").splitlines():
        if line.startswith("0::"):
            rel = line[3:].strip()
    best, limit = None, float("inf")
    path = Path("/sys/fs/cgroup") / rel.lstrip("/")
    while str(path).startswith("/sys/fs/cgroup"):
        text = _read(path / "memory.max").strip()
        if text.isdigit() and int(text) <= limit:
            best, limit = path, int(text)
        if path == Path("/sys/fs/cgroup"):
            break
        path = path.parent
    return best, limit


def _sample(t0, cgroup, limit):
    stat = _read("/proc/self/stat").rsplit(")", 1)[-1].split()      # after "pid (comm)"
    tick = os.sysconf("SC_CLK_TCK")
    page = os.sysconf("SC_PAGE_SIZE")
    mem = _table(_read("/proc/meminfo"), 1024 / GIB)
    numa = []
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        for line in _read(node / "meminfo").splitlines():
            if "MemFree" in line:
                numa.append(f"{float(line.split()[-2]) * 1024 / GIB:.1f}")
    vm = _table(_read("/proc/vmstat"))
    row = {
        "t": f"{time.time() - t0:.1f}",
        "rss": f"{int(stat[21]) * page / GIB:.2f}",
        "utime": f"{int(stat[11]) / tick:.1f}",
        "stime": f"{int(stat[12]) / tick:.1f}",
        "majflt": stat[9],
        "cpu": stat[36],
        "node_free": f"{mem.get('MemFree', 0):.1f}",
        "node_available": f"{mem.get('MemAvailable', 0):.1f}",
        "node_cache": f"{mem.get('Cached', 0):.1f}",
        "node_anon": f"{mem.get('AnonPages', 0):.1f}",
        "numa_free": ",".join(numa) or "-",
        "allocstall": f"{sum(v for k, v in vm.items() if k.startswith('allocstall')):.0f}",
        "compact_stall": f"{vm.get('compact_stall', 0):.0f}",
        "pgscan_direct": f"{vm.get('pgscan_direct', 0):.0f}",
        "pgsteal_direct": f"{vm.get('pgsteal_direct', 0):.0f}",
    }
    if cgroup is not None:
        cg = _table(_read(cgroup / "memory.stat"), 1 / GIB)
        events = _table(_read(cgroup / "memory.events"))
        pressure = {}
        for line in _read(cgroup / "memory.pressure").splitlines():
            kind, *fields = line.split()
            pressure[kind] = dict(f.split("=") for f in fields).get("avg10", "-")
        current = _read(cgroup / "memory.current").strip()
        row.update({
            "cg_limit": f"{limit / GIB:.1f}",
            "cg_use": f"{int(current) / GIB:.1f}" if current.isdigit() else "-",
            "cg_anon": f"{cg.get('anon', 0):.1f}",
            "cg_file": f"{cg.get('file', 0):.1f}",
            "cg_hit_limit": f"{events.get('max', 0):.0f}",
            "cg_some10": pressure.get("some", "-"),
            "cg_full10": pressure.get("full", "-"),
        })
    return "\t".join(row.get(c, "-") for c in COLUMNS)


def start(path, period):
    """Start sampling into ``path`` every ``period`` seconds; returns (stop, mark), where
    mark(text) writes a comment line with the time, to tell the phases of a run apart."""
    cgroup, limit = _tightest_cgroup()
    out = open(path, "w")
    settings = {
        "node": os.uname().nodename,
        "start": time.strftime("%H:%M:%S"),
        "cgroup": str(cgroup),
        "thp": _read("/sys/kernel/mm/transparent_hugepage/enabled").strip(),
        "thp_defrag": _read("/sys/kernel/mm/transparent_hugepage/defrag").strip(),
        "zone_reclaim_mode": _read("/proc/sys/vm/zone_reclaim_mode").strip(),
        "swap_total_kB": str(int(_table(_read("/proc/meminfo")).get("SwapTotal", 0))),
        "memory.high": _read(cgroup / "memory.high").strip() if cgroup else "",
        "memory.swap.max": _read(cgroup / "memory.swap.max").strip() if cgroup else "",
    }
    out.write("# " + "; ".join(f"{k}={v}" for k, v in settings.items()) + "\n")
    out.write("\t".join(COLUMNS) + "\n")
    done = threading.Event()
    t0 = time.time()

    def loop():
        while True:
            try:
                out.write(_sample(t0, cgroup, limit) + "\n")
                out.flush()
            except Exception as error:                    # a diagnostic must not end the run
                out.write(f"# sample failed: {error!r}\n")
            if done.wait(period):
                break

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    def mark(text):
        out.write(f"# t={time.time() - t0:.1f} {text}\n")
        out.flush()

    def stop():
        done.set()
        thread.join(timeout=5)
        out.close()

    return stop, mark

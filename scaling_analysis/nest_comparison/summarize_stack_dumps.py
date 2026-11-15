"""Where every rank is during SuperNeuroABM's first tick, from run_partitioned_sna.py's stack dumps.

    python summarize_stack_dumps.py <root>/stacks_np<N> [--depth 2] [--top 4]

Every rank wrote its Python stack every few seconds (--stack-dump-every). Per dump time this
prints the places the ranks are at, most common first, and how many ranks are at each: the
innermost frames of the thread that is inside simulate(), as file:line function, and the
innermost SAGESim or SuperNeuroABM frame when it is further out. A place that holds a few
ranks only names them and their nodes: a rank the others wait for shows up there.
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

FRAME = re.compile(r'\s+File "(.+?)", line (\d+) in (\S+)')
HEADER = re.compile(r"every (\S+) s from (\S+)(?: on (\S+?); (.*))?")


def dumps_of(path):
    """One rank's file: (period, node, what the header says of the node, per dump the frames,
    innermost first, of the thread inside simulate())."""
    lines = path.read_text().splitlines()
    m = HEADER.match(lines[0]) if lines else None
    period = float(m.group(1)) if m else 0.0
    node, about = (m.group(3) or "?", m.group(4) or "") if m else ("?", "")
    out, threads, frames = [], [], None
    for line in lines[1:] + ["Timeout"]:
        if line.startswith("Timeout"):
            if frames is not None:
                threads.append(frames)
            if threads:
                inside = [t for t in threads if any(f[2] == "simulate" for f in t)]
                out.append(inside[0] if inside else threads[-1])
            threads, frames = [], None
        elif line.startswith(("Thread", "Current thread")):
            if frames is not None:
                threads.append(frames)
            frames = []
        elif frames is not None:
            m = FRAME.match(line)
            if m:
                frames.append((m.group(1), int(m.group(2)), m.group(3)))
    return period, node, about, out


def short(frame):
    file, line, func = frame
    parts = Path(file).parts
    return f"{'/'.join(parts[-2:])}:{line} {func}"


def place(frames, depth):
    """The innermost frames, and the innermost one in SAGESim or SuperNeuroABM if not among them."""
    shown = frames[:depth]
    ours = next((f for f in frames if "sagesim" in f[0].lower() or "superneuroabm/" in f[0]), None)
    if ours is not None and ours not in shown:
        shown = shown + [ours]
    return " < ".join(short(f) for f in shown)


def memory_table(files, show):
    """Per node, from its ranks' memory samples (memory_sampler.py) during the first tick: the
    most its job used against its limit, how often it hit the limit, the worst memory pressure,
    and the system CPU time its ranks spent. The ``show`` nodes that used the most system time."""
    nodes = defaultdict(lambda: defaultdict(float))
    settings = ""
    for f in files:
        lines = f.read_text().splitlines()
        if len(lines) < 3:
            continue
        settings = re.sub(r"node=\S+; |start=\S+; |cgroup=\S+; ", "", lines[0].lstrip("# "))
        node = re.search(r"node=([^;]+)", lines[0]).group(1)
        cols = lines[1].split("\t")
        t_first = t_last = None
        rows = []
        for line in lines[2:]:
            if line.startswith("# t=") and "first tick starts" in line:
                t_first = float(line.split()[1][2:])
            elif line.startswith("# t=") and "first tick ends" in line:
                t_last = float(line.split()[1][2:])
            elif not line.startswith("#"):
                rows.append(dict(zip(cols, line.split("\t"))))
        if t_first is None:
            continue
        tick = [r for r in rows if float(r["t"]) >= t_first and (t_last is None or float(r["t"]) <= t_last)]
        if not tick:
            continue

        def num(row, key):
            try:
                return float(row[key])
            except (KeyError, ValueError):
                return 0.0

        n = nodes[node]
        n["ranks"] += 1
        n["rss_peak"] += max(num(r, "rss") for r in tick)
        n["stime"] += num(tick[-1], "stime") - num(tick[0], "stime")
        n["utime"] += num(tick[-1], "utime") - num(tick[0], "utime")
        n["limit"] = num(tick[0], "cg_limit")
        n["use"] = max(n["use"], max(num(r, "cg_use") for r in tick))
        n["file_min"] = min(num(r, "cg_file") for r in tick)
        n["file_start"] = num(tick[0], "cg_file")
        n["hits"] = max(n["hits"], num(tick[-1], "cg_hit_limit") - num(tick[0], "cg_hit_limit"))
        n["some"] = max(n["some"], max(num(r, "cg_some10") for r in tick))
        n["full"] = max(n["full"], max(num(r, "cg_full10") for r in tick))
        n["available_min"] = min(num(r, "node_available") for r in tick)
        n["allocstall"] = max(n["allocstall"], num(tick[-1], "allocstall") - num(tick[0], "allocstall"))
        n["compact"] = max(n["compact"], num(tick[-1], "compact_stall") - num(tick[0], "compact_stall"))
        n["samples"] += len(tick)
    if not nodes:
        return
    print(f"memory during the first tick, {len(nodes)} nodes ({settings})")
    print("  node        ranks  sum of rank peaks  job use / limit  file cache start > least  hit limit  "
          "pressure some / full (% of 10 s)  least available  sys / user CPU s  reclaim stalls  compaction stalls")
    order = sorted(nodes, key=lambda k: -nodes[k]["stime"])
    for node in order[:show] + (["..."] if len(order) > show + 1 else []) + order[show:][-1:]:
        if node == "...":
            print("  ...")
            continue
        n = nodes[node]
        print(f"  {node:10s}  {n['ranks']:5.0f}  {n['rss_peak']:13.1f} GiB  {n['use']:6.1f} / {n['limit']:5.1f}  "
              f"{n['file_start']:12.1f} > {n['file_min']:5.1f}  {n['hits']:9.0f}  "
              f"{n['some']:15.1f} / {n['full']:5.1f}  {n['available_min']:20.1f}  "
              f"{n['stime']:7.0f} / {n['utime']:6.0f}  {n['allocstall']:14.0f}  {n['compact']:17.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--depth", type=int, default=2, help="innermost frames shown per place")
    ap.add_argument("--top", type=int, default=4, help="places shown per dump time")
    ap.add_argument("--name-up-to", type=int, default=6,
                    help="a place with at most this many ranks lists them and their nodes")
    ap.add_argument("--nodes", type=int, default=6,
                    help="nodes shown in the memory table: those with the most system CPU time")
    args = ap.parse_args()
    memory_table(sorted(Path(args.dir).glob("mem*.txt")), args.nodes)
    files = sorted(Path(args.dir).glob("rank*.txt"))
    ranks = [int(f.stem[4:]) for f in files]
    per_rank = [dumps_of(f) for f in files]
    period = max((p for p, *_ in per_rank), default=0.0)
    most = max((len(d) for *_, d in per_rank), default=0)
    print(f"{len(per_rank)} ranks, a dump every {period:g} s, up to {most} dumps per rank")
    nodes = Counter(re.sub(r"available \d+, page cache \d+", "available .., page cache ..",
                           re.sub(r", free \d+", "", about)) for _, _, about, _ in per_rank)
    for about, count in nodes.most_common():
        print(f"  {count:4d} ranks on: {about}")
    low = sorted((int(m.group(1)), node) for _, node, about, _ in per_rank
                 for m in [re.search(r"available (\d+)", about)] if m)
    if low:
        print(f"  available node memory at the start of the tick: {low[0][0]} GiB ({low[0][1]}) "
              f"to {low[-1][0]} GiB")
    for k in range(most):
        at = defaultdict(list)
        for rank, (_, node, _, d) in zip(ranks, per_rank):
            if len(d) > k:
                at[place(d[k], args.depth)].append((rank, node))
        print(f"{(k + 1) * period:7.1f} s  ({sum(map(len, at.values()))} ranks still in the first tick)")
        for where, who in sorted(at.items(), key=lambda item: -len(item[1]))[:args.top]:
            named = ""
            if len(who) <= args.name_up_to:
                named = "   [" + ", ".join(f"rank {r} on {n}" for r, n in who) + "]"
            print(f"    {len(who):4d}  {where}{named}")


if __name__ == "__main__":
    main()

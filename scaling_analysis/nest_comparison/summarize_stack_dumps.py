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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--depth", type=int, default=2, help="innermost frames shown per place")
    ap.add_argument("--top", type=int, default=4, help="places shown per dump time")
    ap.add_argument("--name-up-to", type=int, default=6,
                    help="a place with at most this many ranks lists them and their nodes")
    args = ap.parse_args()
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

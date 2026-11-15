"""Where every rank is during SuperNeuroABM's first tick, from run_partitioned_sna.py's stack dumps.

    python summarize_stack_dumps.py <root>/stacks_np<N> [--depth 3] [--top 4]

Every rank wrote its Python stack every few seconds (--stack-dump-every). Per dump time this
prints the places the ranks are at, most common first, and how many ranks are at each: the
innermost frames of the thread that is inside simulate(), as file:line function, and the
innermost SAGESim or SuperNeuroABM frame when it is further out.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

FRAME = re.compile(r'\s+File "(.+?)", line (\d+) in (\S+)')


def dumps_of(path):
    """Per dump of one rank: the frames (innermost first) of the thread inside simulate()."""
    lines = path.read_text().splitlines()
    period = float(lines[0].split()[1]) if lines and lines[0].startswith("every") else 0.0
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
    return period, out


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
    args = ap.parse_args()
    per_rank = [dumps_of(p) for p in sorted(Path(args.dir).glob("rank*.txt"))]
    period = max((p for p, _ in per_rank), default=0.0)
    most = max((len(d) for _, d in per_rank), default=0)
    print(f"{len(per_rank)} ranks, a dump every {period:g} s, up to {most} dumps per rank")
    for k in range(most):
        places = Counter(place(d[k], args.depth) for _, d in per_rank if len(d) > k)
        print(f"{(k + 1) * period:7.1f} s  ({sum(places.values())} ranks still in the first tick)")
        for where, count in places.most_common(args.top):
            print(f"    {count:4d}  {where}")


if __name__ == "__main__":
    main()

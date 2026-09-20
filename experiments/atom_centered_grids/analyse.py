"""Points needed to reach a given MP2 accuracy, one column per fitting mode.

Ratios are taken against the leftmost mode the file holds, which is the most compact
grid any of these modes produces and so the natural denominator: `blocked` prices giving
up molecular pruning, the `_orbits` modes price giving up point-wise pruning on top of
it (see orbits.py), and `free` / `ghost` price giving up the molecule altogether
(ghosts.py). A file from ghosts.py has no `global` column, so there the denominator is
`blocked` - which is the right comparison for it anyway, since what an offline fit gives
up is exactly what the blocked fit still has.
"""
import json, glob, sys
import numpy as np

TARGETS = [50.0, 20.0, 10.0, 5.0, 2.0]   # uHa above the converged (full-Becke) value


def curve(rows, mode):
    pts = [(r['n_points'], r['err_uha']) for r in rows
           if r.get('mode') == mode and 'n_points' in r]
    return sorted(pts)


def points_for(pts, floor, target):
    """Interpolate n_points at which |err - floor| first drops to target (log-log)."""
    xs, ys = [], []
    for n, e in pts:
        d = abs(e - floor)
        if d > 1e-9:
            xs.append(n); ys.append(d)
    if not xs:
        return None
    xs, ys = np.array(xs, float), np.array(ys, float)
    if ys.min() > target:
        return None                       # never got that accurate
    if ys[0] <= target:
        return xs[0]                      # already there at the loosest grid
    for i in range(len(xs) - 1):
        if ys[i] > target >= ys[i + 1]:
            lx = np.interp(np.log(target), [np.log(ys[i + 1]), np.log(ys[i])],
                           [np.log(xs[i + 1]), np.log(xs[i])])
            return float(np.exp(lx))
    return None


def main(paths):
    for path in paths or sorted(glob.glob('*.json')):
        d = json.load(open(path))
        if 'rows' not in d:
            continue
        meta, rows = d['meta'], d['rows']
        becke = next((r for r in rows if r.get('mode') == 'becke'), None)
        if becke is None:
            continue
        floor = becke['err_uha']              # converged value on the full parent grid
        # Whichever modes this file actually holds, in a fixed order. Anything the fixed
        # order does not name - ensemble.py labels its rows by ensemble, not by mode - is
        # appended in the order it first appears, so a file of unknown modes still ranks
        # against whichever of them came first.
        known = ('global', 'blocked', 'global_orbits', 'blocked_orbits', 'free', 'ghost')
        modes = [m for m in known if curve(rows, m)]
        for r in rows:
            m = r.get('mode')
            if m and m != 'becke' and m not in known and m not in modes:
                modes.append(m)
        curves = {m: curve(rows, m) for m in modes}

        print(f"\n=== {meta['molecule']}: {meta['natm']} atoms, {meta['nao']} AOs "
              f"(parent grid {becke['n_points']} pts, converged err {floor:+.2f} uHa)")
        header = f"  {'target':>9s}" + "".join(f" {m:>14s}" for m in modes)
        print(header)
        for t in TARGETS:
            need = {m: points_for(curves[m], floor, t) for m in modes}
            base = need.get(modes[0])
            cells = []
            for m in modes:
                if need[m] is None:
                    cells.append(f" {'n/a':>14s}")
                elif base and m != modes[0]:
                    cells.append(f" {need[m]:8.0f} ({need[m] / base:4.2f}x)")
                else:
                    cells.append(f" {need[m]:14.0f}")
            print(f"  {t:7.0f}uHa" + "".join(cells))

if __name__ == '__main__':
    main(sys.argv[1:])

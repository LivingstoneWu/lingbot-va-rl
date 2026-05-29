"""
Plot gradient alignment statistics from grad_stats.jsonl.

Each record in the file was written by Trainer._write_grad_stats and has the form:
  {
    "step": 200,
    "t_latent": 612.3,
    "t_action": 448.1,
    "blocks": {
      "0":  {"mag_latent": 0.041, "mag_action": 0.027, "cos_latent_action": 0.18},
      "1":  { ... },
      ...
      "-1": { ... }   <- non-block parameters (embeddings, final norm, ...)
    }
  }

Usage examples
--------------
# Cosine similarity (all pairs) across all blocks, steps 0-2000:
python wan_va/plot_grad_stats.py \\
    --log-file checkpoints/grad_stats.jsonl \\
    --step-range 0 2000 \\
    --plot-type cosine \\
    --output figs/cos_early.png

# Cosine similarity for a specific pair only:
python wan_va/plot_grad_stats.py \\
    --log-file checkpoints/grad_stats.jsonl \\
    --plot-type cosine \\
    --ratio-pairs latent/action \\
    --output figs/cos_lat_act.png

# Magnitude ratio (all pairs) over full training:
python wan_va/plot_grad_stats.py \\
    --log-file checkpoints/grad_stats.jsonl \\
    --plot-type magnitude-ratio \\
    --output figs/mag_ratio.png

# Magnitude ratio for a specific pair:
python wan_va/plot_grad_stats.py \\
    --log-file checkpoints/grad_stats.jsonl \\
    --plot-type magnitude-ratio \\
    --ratio-pairs latent/action \\
    --output figs/mag_ratio_lat_act.png

# Absolute magnitude evolution, latent term only:
python wan_va/plot_grad_stats.py \\
    --log-file checkpoints/grad_stats.jsonl \\
    --plot-type magnitude \\
    --term latent \\
    --output figs/mag_latent.png

# Cosine binned by denoising timestep (t_latent), late training:
python wan_va/plot_grad_stats.py \\
    --log-file checkpoints/grad_stats.jsonl \\
    --step-range 8000 10000 \\
    --plot-type cosine-vs-t \\
    --pair latent action \\
    --t-bins 10 \\
    --output figs/cos_vs_t_late.png
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(log_file: str, step_min: int, step_max: int) -> list:
    records = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec.get('step', -1)
            if step_min <= s <= step_max:
                records.append(rec)
    return records


def collect_block_ids(records: list) -> list:
    """Return sorted numbered block ids (ints ≥ 0); -1 ('other') excluded."""
    ids = set()
    for rec in records:
        for k in rec.get('blocks', {}):
            v = int(k)
            if v >= 0:
                ids.add(v)
    return sorted(ids)


def collect_pairs(records: list) -> list:
    """
    Return list of (term_a, term_b) cosine pairs in the order they were written,
    inferred directly from 'cos_*' keys present in the data.
    Falls back to alphabetically-derived pairs if no cosine keys exist yet.
    """
    seen = {}  # (ta, tb) -> True, preserving first-seen order
    for rec in records:
        for entry in rec.get('blocks', {}).values():
            for key in entry:
                if key.startswith('cos_'):
                    rest = key[4:]  # e.g. "latent_action"
                    # Split on first underscore that produces two known terms.
                    # Terms can themselves contain underscores (unlikely but safe).
                    terms_all = collect_terms(records)
                    for ta in terms_all:
                        prefix = ta + '_'
                        if rest.startswith(prefix):
                            tb = rest[len(prefix):]
                            if tb in terms_all:
                                pair = (ta, tb)
                                if pair not in seen:
                                    seen[pair] = True
                                break
    if seen:
        return list(seen.keys())
    # Fallback: derive from sorted terms
    terms = collect_terms(records)
    return [(ta, tb) for i, ta in enumerate(terms) for tb in terms[i + 1:]]


def collect_terms(records: list) -> list:
    terms = set()
    for rec in records:
        for bid_str, entry in rec.get('blocks', {}).items():
            for key in entry:
                if key.startswith('mag_'):
                    terms.add(key[4:])
    return sorted(terms)


# ---------------------------------------------------------------------------
# Colormap helper  (spectrum: shallow=blue → deep=red via turbo)
# ---------------------------------------------------------------------------

def _block_cmap(block_ids: list):
    """
    Return a dict {block_id: colour} using the 'turbo' spectrum so that
    shallow blocks are blue/purple and deep blocks are yellow/red.
    Block -1 ('other') is always plotted in grey with a dashed linestyle.
    """
    import matplotlib
    cmap = matplotlib.colormaps['turbo']
    n = max(len(block_ids), 1)
    return {bid: cmap(i / (n - 1) if n > 1 else 0.5)
            for i, bid in enumerate(block_ids)}


def _add_depth_colorbar(fig, block_ids: list, label: str = 'Block depth (shallow → deep)'):
    """Append a thin colorbar on the right representing layer depth."""
    import matplotlib
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    if not block_ids:
        return
    cmap  = matplotlib.colormaps['turbo']
    norm  = mcolors.Normalize(vmin=min(block_ids), vmax=max(block_ids))
    sm    = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar  = fig.colorbar(sm, ax=fig.get_axes(), shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(label, fontsize=8)


# ---------------------------------------------------------------------------
# Plot: magnitude-ratio over training steps
# ---------------------------------------------------------------------------

def plot_magnitude_ratio(records, ratio_pairs, block_ids, output, smooth: int = 1):
    """
    For each ratio pair (term_a, term_b), plot  ‖∇term_a‖ / ‖∇term_b‖  per
    block over training steps.  One subplot per pair, blocks coloured by depth.
    A horizontal dashed line at y=1 marks equal magnitude.

    smooth: moving-average window size (1 = no smoothing).
    """
    import matplotlib.pyplot as plt

    n_pairs = len(ratio_pairs)
    if n_pairs == 0:
        print("No ratio pairs to plot.")
        return

    colours  = _block_cmap(block_ids)
    fig, axes = plt.subplots(
        n_pairs, 1,
        figsize=(11, 3.2 * n_pairs),
        sharex=True,
        squeeze=False,
    )

    step_range_str = (
        f"steps {min(r['step'] for r in records)}–{max(r['step'] for r in records)}"
    )
    smooth_str = f'  [smooth={smooth}]' if smooth > 1 else ''

    for ax_row, (ta, tb) in zip(axes, ratio_pairs):
        ax   = ax_row[0]
        ka   = f'mag_{ta}'   # magnitude keys are unordered — always present
        kb   = f'mag_{tb}'

        for bid in block_ids:
            bid_str = str(bid)
            steps_  = []
            ratios_ = []
            for rec in records:
                entry = rec.get('blocks', {}).get(bid_str, {})
                va    = entry.get(ka)
                vb    = entry.get(kb)
                if va is not None and vb is not None and vb > 1e-12:
                    steps_.append(rec['step'])
                    ratios_.append(va / vb)
            if not steps_:
                continue
            colour = colours[bid]
            if smooth > 1:
                ax.plot(steps_, ratios_, color=colour,
                        linewidth=0.5, alpha=0.2)
                s_steps, s_ratios = _smooth(steps_, ratios_, smooth)
                ax.plot(s_steps, s_ratios, color=colour,
                        linewidth=1.2, alpha=0.9)
            else:
                ax.plot(steps_, ratios_, color=colour,
                        linewidth=0.9, alpha=0.85)

        ax.axhline(1.0, color='k', linewidth=0.6, linestyle='--', alpha=0.5,
                   label='ratio = 1')
        ax.set_ylabel(f'‖∇{ta}‖ / ‖∇{tb}‖', fontsize=9)
        ax.set_title(f'{ta} / {tb}  [{step_range_str}]{smooth_str}', fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_yscale('log')   # log scale makes ratios above/below 1 symmetric

    axes[-1][0].set_xlabel('Optimizer step')
    fig.suptitle('Per-block gradient magnitude ratios', fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    _add_depth_colorbar(fig, block_ids)
    _save_or_show(fig, output)


# ---------------------------------------------------------------------------
# Plot: absolute magnitude evolution over training steps
# ---------------------------------------------------------------------------

def plot_magnitude_evolution(records, terms, block_ids, output, smooth: int = 1):
    import matplotlib.pyplot as plt

    colours    = _block_cmap(block_ids)
    smooth_str = f'  [smooth={smooth}]' if smooth > 1 else ''
    fig, axes  = plt.subplots(
        len(terms), 1,
        figsize=(11, 3 * len(terms)),
        sharex=True,
        squeeze=False,
    )

    step_range_str = (
        f"steps {min(r['step'] for r in records)}–{max(r['step'] for r in records)}"
    )

    for ax_row, term in zip(axes, terms):
        ax  = ax_row[0]
        key = f'mag_{term}'
        for bid in block_ids:
            bid_str = str(bid)
            steps_ = [rec['step'] for rec in records
                      if bid_str in rec.get('blocks', {})]
            vals_  = [rec['blocks'][bid_str].get(key, float('nan'))
                      for rec in records
                      if bid_str in rec.get('blocks', {})]
            if not steps_:
                continue
            colour = colours[bid]
            if smooth > 1:
                ax.plot(steps_, vals_, color=colour, linewidth=0.5, alpha=0.2)
                s_steps, s_vals = _smooth(steps_, vals_, smooth)
                ax.plot(s_steps, s_vals, color=colour, linewidth=1.2, alpha=0.9)
            else:
                ax.plot(steps_, vals_, color=colour, linewidth=0.9, alpha=0.85)

        # also show 'other' (block -1) params in grey dashed if present
        if any('-1' in rec.get('blocks', {}) for rec in records):
            steps_ = [rec['step'] for rec in records if '-1' in rec.get('blocks', {})]
            vals_  = [rec['blocks']['-1'].get(key, float('nan'))
                      for rec in records if '-1' in rec.get('blocks', {})]
            if smooth > 1:
                ax.plot(steps_, vals_, color='grey', linewidth=0.4,
                        linestyle='--', alpha=0.15)
                s_steps, s_vals = _smooth(steps_, vals_, smooth)
                ax.plot(s_steps, s_vals, color='grey', linewidth=0.9,
                        linestyle='--', alpha=0.6, label='other')
            else:
                ax.plot(steps_, vals_, color='grey', linewidth=0.7,
                        linestyle='--', alpha=0.6, label='other')

        ax.set_ylabel(f'‖∇{term}‖', fontsize=9)
        ax.set_title(f'{term}  [{step_range_str}]{smooth_str}', fontsize=9)
        ax.grid(alpha=0.25)

    axes[-1][0].set_xlabel('Optimizer step')
    fig.suptitle('Per-block gradient magnitudes', fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    _add_depth_colorbar(fig, block_ids)
    _save_or_show(fig, output)


# ---------------------------------------------------------------------------
# Plot: cosine similarity over training steps, blocks coloured by depth
# ---------------------------------------------------------------------------

def plot_cosine_distribution(records, pairs, block_ids, output, smooth: int = 1):
    """
    For each cosine pair, plot the per-block cosine similarity over training
    steps.  One subplot per pair; blocks coloured by depth via the turbo
    spectrum (shallow=blue, deep=red).  A dashed line at y=0 marks the
    gradient-orthogonality boundary.

    smooth: moving-average window size (1 = no smoothing).  When > 1 the raw
            series is plotted in low-alpha behind the smoothed line.
    """
    import matplotlib.pyplot as plt

    n_pairs = len(pairs)
    if n_pairs == 0:
        print("No cosine pairs to plot.")
        return

    colours = _block_cmap(block_ids)
    fig, axes = plt.subplots(
        n_pairs, 1,
        figsize=(11, 3.2 * n_pairs),
        sharex=True,
        squeeze=False,
    )

    step_range_str = (
        f"steps {min(r['step'] for r in records)}–{max(r['step'] for r in records)}"
    )
    smooth_str = f'  [smooth={smooth}]' if smooth > 1 else ''

    for ax_row, (ta, tb) in zip(axes, pairs):
        ax  = ax_row[0]
        # Try both orderings — the stored key depends on insertion order in train.py
        key      = f'cos_{ta}_{tb}'
        key_flip = f'cos_{tb}_{ta}'

        for bid in block_ids:
            bid_str = str(bid)
            steps_ = []
            vals_  = []
            for rec in records:
                entry = rec.get('blocks', {}).get(bid_str, {})
                v = entry.get(key)
                if v is None:
                    v = entry.get(key_flip)
                if v is not None:
                    steps_.append(rec['step'])
                    vals_.append(v)
            if not steps_:
                continue
            colour = colours[bid]
            if smooth > 1:
                ax.plot(steps_, vals_, color=colour,
                        linewidth=0.5, alpha=0.2)
                s_steps, s_vals = _smooth(steps_, vals_, smooth)
                ax.plot(s_steps, s_vals, color=colour,
                        linewidth=1.2, alpha=0.9)
            else:
                ax.plot(steps_, vals_, color=colour,
                        linewidth=0.9, alpha=0.85)

        ax.axhline(0.0, color='k', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_ylabel(f'cos({ta}, {tb})', fontsize=9)
        ax.set_title(f'{ta} vs {tb}  [{step_range_str}]{smooth_str}', fontsize=9)
        ax.grid(alpha=0.25)

    axes[-1][0].set_xlabel('Optimizer step')
    fig.suptitle('Per-block gradient cosine similarity', fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    _add_depth_colorbar(fig, block_ids)
    _save_or_show(fig, output)


# ---------------------------------------------------------------------------
# Plot: cosine vs denoising timestep
# ---------------------------------------------------------------------------

def plot_cosine_vs_t(records, pair, t_bins, t_field, output):
    import matplotlib.pyplot as plt
    ta, tb   = pair
    cos_key  = f'cos_{ta}_{tb}'
    t_values = np.array([r[t_field] for r in records])
    t_min, t_max = t_values.min(), t_values.max()
    edges       = np.linspace(t_min, t_max, t_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_indices = np.clip(np.digitize(t_values, edges) - 1, 0, t_bins - 1)

    bin_cos: defaultdict = defaultdict(list)
    for i, rec in enumerate(records):
        b = bin_indices[i]
        for bid_str, entry in rec.get('blocks', {}).items():
            if cos_key in entry:
                bin_cos[b].append(entry[cos_key])

    bin_means = [float(np.mean(bin_cos[b])) if bin_cos[b] else float('nan')
                 for b in range(t_bins)]
    bin_stds  = [float(np.std(bin_cos[b]))  if bin_cos[b] else float('nan')
                 for b in range(t_bins)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_centers, bin_means,
           width=(t_max - t_min) / t_bins * 0.8,
           yerr=bin_stds, capsize=3, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel(t_field)
    ax.set_ylabel(f'mean cos({ta}, {tb}) across blocks')
    step_range_str = (
        f"steps {min(r['step'] for r in records)}–{max(r['step'] for r in records)}"
    )
    ax.set_title(f'Gradient cosine vs timestep: {ta} vs {tb}  [{step_range_str}]')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(steps: list, vals: list, window: int):
    """
    Apply a centred uniform moving-average with the given window size.
    Returns (steps_out, smooth_vals).  If window <= 1 the input is returned
    unchanged.  NaN values are ignored (filled with surrounding mean).
    """
    if window <= 1 or len(vals) < window:
        return steps, vals
    arr = np.array(vals, dtype=float)
    # Use a cumsum-based O(n) sliding mean; pad edges with NaN then fill.
    half  = window // 2
    padded = np.pad(arr, (half, half), constant_values=np.nan)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        window_vals = padded[i: i + window]
        valid = window_vals[~np.isnan(window_vals)]
        if valid.size:
            out[i] = valid.mean()
    return steps, out.tolist()


def _save_or_show(fig, output):
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def _parse_ratio_pairs(raw: list[str], all_pairs: list) -> list:
    """
    Parse --ratio-pairs arguments of the form 'term_a/term_b'.
    Returns a list of (term_a, term_b) tuples.
    Falls back to all_pairs if raw is empty.
    """
    if not raw:
        return all_pairs
    out = []
    for s in raw:
        parts = s.split('/')
        if len(parts) != 2:
            print(f"Warning: ignoring malformed ratio pair '{s}' (expected 'A/B')")
            continue
        out.append((parts[0].strip(), parts[1].strip()))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot gradient alignment statistics from grad_stats.jsonl'
    )
    parser.add_argument('--log-file', required=True,
                        help='Path to grad_stats.jsonl')
    parser.add_argument('--step-range', nargs=2, type=int, default=[0, int(1e9)],
                        metavar=('MIN', 'MAX'),
                        help='Inclusive step range to analyse (default: all)')
    parser.add_argument('--plot-type', default='magnitude-ratio',
                        choices=['cosine', 'magnitude', 'magnitude-ratio', 'cosine-vs-t'],
                        help='Which plot to generate (default: magnitude-ratio)')
    parser.add_argument('--pair', nargs=2, default=['latent', 'action'],
                        metavar=('TERM_A', 'TERM_B'),
                        help='Loss pair for cosine / cosine-vs-t plots (default: latent action)')
    parser.add_argument('--ratio-pairs', nargs='+', default=[],
                        metavar='A/B',
                        help='Ratio pairs for magnitude-ratio plot, e.g. latent/action '
                             '(default: all pairs found in data)')
    parser.add_argument('--term', default=None,
                        help='Single loss term for magnitude plot (default: all terms)')
    parser.add_argument('--blocks', nargs='+', type=int, default=None,
                        help='Block ids to show (default: all numbered blocks)')
    parser.add_argument('--t-bins', type=int, default=10,
                        help='Number of timestep bins for cosine-vs-t plot')
    parser.add_argument('--t-field', default='t_latent',
                        choices=['t_latent', 't_action'],
                        help='Timestep field to bin by (default: t_latent)')
    parser.add_argument('--smooth', type=int, default=1, metavar='W',
                        help='Uniform moving-average window size for line plots '
                             '(default: 1 = no smoothing).  When > 1 the raw '
                             'series is shown faintly behind the smoothed line.')
    parser.add_argument('--output', default=None,
                        help='Output image path (show interactively if omitted)')
    args = parser.parse_args()

    records = load_records(args.log_file, args.step_range[0], args.step_range[1])
    if not records:
        print(f'No records found in [{args.step_range[0]}, {args.step_range[1]}]')
        sys.exit(1)
    print(f'Loaded {len(records)} records, '
          f'steps {records[0]["step"]}–{records[-1]["step"]}')

    block_ids = collect_block_ids(records)  # numbered blocks only (≥ 0)
    bids      = args.blocks if args.blocks is not None else block_ids

    if args.plot_type == 'magnitude-ratio':
        all_pairs   = collect_pairs(records)
        ratio_pairs = _parse_ratio_pairs(args.ratio_pairs, all_pairs)
        if not ratio_pairs:
            print("No ratio pairs found in data.")
            sys.exit(1)
        plot_magnitude_ratio(records, ratio_pairs, bids, args.output,
                             smooth=args.smooth)

    elif args.plot_type == 'magnitude':
        terms = [args.term] if args.term else collect_terms(records)
        plot_magnitude_evolution(records, terms, bids, args.output,
                                 smooth=args.smooth)

    elif args.plot_type == 'cosine':
        all_pairs    = collect_pairs(records)
        cosine_pairs = _parse_ratio_pairs(args.ratio_pairs, all_pairs)
        if not cosine_pairs:
            print("No cosine pairs found in data.")
            sys.exit(1)
        plot_cosine_distribution(records, cosine_pairs, bids, args.output,
                                 smooth=args.smooth)

    elif args.plot_type == 'cosine-vs-t':
        plot_cosine_vs_t(
            records, tuple(args.pair), args.t_bins, args.t_field, args.output
        )


if __name__ == '__main__':
    main()

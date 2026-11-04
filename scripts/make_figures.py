"""Render every figure in docs/figures from the data in benchmarks/.

  python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
B = os.path.join(ROOT, "benchmarks")
os.makedirs(OUT, exist_ok=True)

# ---- design tokens (light surface) -----------------------------------------------------------
SURF, INK, INK2, MUTED, GRID, CELL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#eceae4"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}            # atoms per matrix (ordinal blue ramp)
PL = BLUE
METHOD_COLOR = {"PursuitLoRA": BLUE, "prompt": ORANGE, "LayerNorm": AQUA, "LoRA": YELLOW, "(IA)": MAGENTA}
ROLE_GROUP = [("attention v", BLUE), ("attention output", ORANGE), ("MLP in", AQUA),
              ("attention q, k", YELLOW), ("MLP out", MAGENTA)]
ROLES = ["attn.c_attn.q", "attn.c_attn.k", "attn.c_attn.v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
ROLE_LABEL = ["q", "k", "v", "o", "MLP in", "MLP out"]

plt.rcParams.update({
    "font.family": "Inter", "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "semibold",
    "axes.labelsize": 12, "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlepad": 14, "legend.frameon": False,
})


def role_group(role):
    if role.endswith("c_attn.v"):
        return 0
    if role == "attn.c_proj":
        return 1
    if role == "mlp.c_fc":
        return 2
    if role.startswith("attn.c_attn"):
        return 3
    return 4


def color_of(name):
    for k, c in METHOD_COLOR.items():
        if k.lower() in name.lower():
            return c
    return MUTED


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec=None, tc=INK, fs=12, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, weight=weight)


def arrow(ax, x0, y0, x1, y1, c=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=c, lw=1.6))


R = json.load(open(os.path.join(B, "results.json")))
S = json.load(open(os.path.join(B, "descent_spectrum_gpt2_bengali.json")))
BASE = R["setup"]["base_val"]
TIERS = R["tiers_seed0"]


def sel_grid(sel):
    """12 x 6 array of atom counts from a selection dict {'L3:attn.c_attn.v': 1, ...}."""
    g = np.zeros((12, 6), dtype=int)
    for k, n in sel.items():
        layer, role = k.split(":", 1)
        g[int(layer[1:]), ROLES.index(role)] = n
    return g


def draw_grid(ax, g, cell=0.62, x0=0.0, y0=0.0, highlight=None, show_roles=True, show_layers=True, fs=10):
    for li in range(12):
        for ri in range(6):
            n = g[li, ri]
            fc = SEQ[min(n, 3)] if n else CELL
            if highlight is not None and highlight[li, ri]:
                fc = YELLOW
            ax.add_patch(Rectangle((x0 + ri * cell, y0 + (11 - li) * cell), cell * 0.9, cell * 0.9, fc=fc, ec="none"))
            if n > 1:
                ax.text(x0 + ri * cell + cell * 0.45, y0 + (11 - li) * cell + cell * 0.45, str(n), ha="center",
                        va="center", color="white", fontsize=fs - 1, weight="semibold")
    if show_roles:
        for ri, r in enumerate(ROLE_LABEL):
            ax.text(x0 + ri * cell + cell * 0.45, y0 + 12 * cell + 0.08, r, ha="center", va="bottom",
                    color=INK2, fontsize=fs, rotation=0 if len(r) < 3 else 35)
    if show_layers:
        for li in range(12):
            ax.text(x0 - 0.12, y0 + (11 - li) * cell + cell * 0.45, f"{li}", ha="right", va="center",
                    color=MUTED, fontsize=fs - 1)


# ---- 1. LoRA vs PursuitLoRA concept --------------------------------------------------------------
def fig_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    for ax in axes:
        ax.set_xlim(-0.9, 5.2)
        ax.set_ylim(-1.6, 8.6)
        ax.set_aspect("equal")
        ax.axis("off")
    hl = np.zeros((12, 6), dtype=bool)
    hl[:, [0, 2]] = True
    axes[0].set_title("LoRA: you choose the matrices and the rank", color=INK)
    draw_grid(axes[0], np.zeros((12, 6), dtype=int), highlight=hl)
    axes[0].text(1.85, -0.35, "rank 1 on q and v of all 12 layers\n24 matrices, the same rank in each\n36,864 parameters",
                 ha="center", va="top", color=INK2, fontsize=11)
    axes[1].set_title("PursuitLoRA: the gradient chooses", color=INK)
    g = sel_grid(R["selections"]["~37k"]["sigma"])
    draw_grid(axes[1], g)
    n_atoms, n_mat = int(g.sum()), int((g > 0).sum())
    axes[1].text(1.85, -0.35, f"same budget (≈37k): {n_atoms} atoms in {n_mat} of 72 matrices\n"
                              f"rank 0 to {int(g.max())}, wherever the loss falls fastest\n36,112 parameters",
                 ha="center", va="top", color=INK2, fontsize=11)
    for ax in axes:
        ax.text(-0.75, 3.7, "layer", rotation=90, ha="center", va="center", color=MUTED, fontsize=10)
    h = [Rectangle((0, 0), 1, 1, fc=c) for c in (YELLOW, SEQ[1], SEQ[2], CELL)]
    fig.legend(h, ["LoRA pair", "1 atom", "2 atoms", "not adapted"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=11, handlelength=1.1)
    save(fig, "01_lora_vs_pursuitlora.png")


# ---- 2. workflow ---------------------------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [("1  Wrap", "get_pursuitlora_model\n(nothing changes yet)"),
             ("2  Probe", "one backward pass\non 4 batches"),
             ("3  Pick atoms", "steepest directions\nacross all matrices,\nuntil the budget is full"),
             ("4  Train", "AdamW or Trainer,\njust like LoRA"),
             ("5  Merge / share", "merge_and_unload()\nor push_to_hub()")]
    for i, (t, s) in enumerate(steps):
        x = 0.15 + i * 2.4
        hi = i in (1, 2)
        box(ax, x, 1.75, 2.0, 0.85, t, PL if hi else "#eceae4", tc="white" if hi else INK, fs=13, weight="semibold")
        ax.text(x + 1.0, 1.45, s, ha="center", va="top", color=INK2, fontsize=10.5)
        if i < 4:
            arrow(ax, x + 2.03, 2.17, x + 2.37, 2.17)
    ax.text(0.15 + 2.4, 0.08, "the only steps LoRA does not have", color=PL, fontsize=11, weight="semibold")
    save(fig, "02_workflow.png")


# ---- 3. anatomy of an atom -----------------------------------------------------------------------
def fig_atom():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.4, 0.6), 3.4, 3.4, fc=CELL, ec=MUTED, lw=0.8))
    ax.text(2.1, 2.3, "frozen\nweight matrix", ha="center", va="center", color=INK2, fontsize=12)
    ax.text(4.35, 2.3, "+", ha="center", va="center", fontsize=26, color=INK2)
    ax.add_patch(Rectangle((5.0, 2.05), 0.5, 0.5, fc=ORANGE, ec="none"))
    ax.text(5.25, 1.75, "strength\n(1 number,\nstarts at 0)", ha="center", va="top", color=INK, fontsize=10.5)
    ax.text(5.85, 2.3, "×", ha="center", va="center", fontsize=20, color=INK2)
    ax.add_patch(Rectangle((6.3, 0.6), 0.32, 3.4, fc=PL, ec="none"))
    ax.text(6.85, 2.3, "input\ndirection", ha="left", va="center", color=INK, fontsize=10.5)
    ax.add_patch(Rectangle((8.2, 3.68), 3.4, 0.32, fc=PL, ec="none"))
    ax.text(9.9, 4.25, "output direction", ha="center", va="bottom", color=INK, fontsize=10.5)
    ax.text(8.2, 3.1, "One atom = the same shape as one LoRA rank.\n\n"
                      "Attention matrix (768 × 768):\n768 + 768 + 1 = 1,537 parameters\n\n"
                      "MLP matrix (768 × 3,072):\n768 + 3,072 + 1 = 3,841 parameters",
            ha="left", va="top", color=INK2, fontsize=10.5)
    save(fig, "03_atom.png")


# ---- 4. descent spectrum ---------------------------------------------------------------------------
def greedy(cands, budget):
    used, out = 0, []
    for j, c in sorted(enumerate(cands), key=lambda t: -t[1][0]):
        if used + c[2] <= budget:
            out.append(j)
            used += c[2]
    return set(out)


def fig_spectrum():
    cands = []
    for m in S["matrices"]:
        for s in m["sigma_top8"]:
            cands.append((s, role_group(m["role"]), m["cost"]))
    order = sorted(range(len(cands)), key=lambda j: -cands[j][0])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    xs = np.arange(1, len(order) + 1)
    top = 80
    for gi, (lab, c) in enumerate(ROLE_GROUP):
        idx = [i for i, j in enumerate(order[:top]) if cands[j][1] == gi]
        ax.scatter(xs[idx], [cands[order[i]][0] for i in idx], s=46, color=c, label=lab, zorder=3,
                   edgecolor=SURF, linewidth=1.2)
    for budget, lab in [(9216, "≈9k budget"), (36864, "≈37k"), (55296, "≈55k")]:
        chosen = greedy(cands, budget)
        last = max(i for i, j in enumerate(order) if j in chosen) + 1
        ax.axvline(last + 0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(last + 1.2, ax.get_ylim()[1] if False else cands[order[0]][0] * 0.98, f"{lab}\n{len(chosen)} atoms",
                color=INK2, fontsize=10, va="top")
    ax.set_xlim(0, top + 1)
    ax.set_xlabel("candidate atoms, steepest first (top 80 of 576)")
    ax.set_ylabel("initial descent rate")
    ax.set_title("A few directions are much steeper than the rest", color=INK)
    grid(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.72), fontsize=10.5, handletextpad=0.3)
    save(fig, "04_descent_spectrum.png")


# ---- 5. where the atoms go ---------------------------------------------------------------------------
def fig_where():
    fig, axes = plt.subplots(1, 4, figsize=(13, 6.4))
    for ax, tier in zip(axes, ["~1.5k", "~9k", "~37k", "~55k"]):
        ax.set_xlim(-0.7, 3.9)
        ax.set_ylim(-0.4, 8.6)
        ax.set_aspect("equal")
        ax.axis("off")
        g = sel_grid(R["selections"][tier]["sigma"])
        draw_grid(ax, g, show_layers=(tier == "~1.5k"), fs=9.5)
        ax.text(1.85, -0.2, f"{int(g.sum())} atom{'s' if g.sum() > 1 else ''}, {int((g > 0).sum())} matri{'ces' if (g > 0).sum() > 1 else 'x'}", ha="center", va="top",
                color=INK2, fontsize=10.5)
        ax.text(0.0, 8.35, f"budget {tier.replace('~', '≈')}", ha="left", color=INK, fontsize=13, weight="semibold")
    h = [Rectangle((0, 0), 1, 1, fc=c) for c in (SEQ[1], SEQ[2], SEQ[3], CELL)]
    fig.legend(h, ["1 atom", "2 atoms", "3 atoms", "not adapted"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), fontsize=11, handlelength=1.1)
    fig.suptitle("Where the atoms go (GPT-2, Bengali): rows are layers 0–11", x=0.06, ha="left", y=0.98,
                 fontsize=15, weight="semibold")
    save(fig, "05_where_atoms_go.png")


# ---- 6. training curves ---------------------------------------------------------------------------
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)
    for ax, tier in zip(axes, ["~9k", "~37k"]):
        rows = [r for r in TIERS[tier] if "ols" not in r["method"] and "QLoRA" not in r["method"]
                and "VeRA" not in r["method"]]
        ends = []
        for r in rows:
            st, v = zip(*r["curve"])
            c = color_of(r["method"])
            pl = "PursuitLoRA" in r["method"]
            ax.plot(st[1:], v[1:], color=c, lw=2.6 if pl else 2, zorder=4 if pl else 3,
                    marker="o", ms=8 if pl else 6.5, mec=SURF, mew=1.5)
            ends.append([v[-1], r["method"].replace(" (sigma)", "").replace("(IA)3", "(IA)³").replace(", NF4 base", ""), c, r["params"]])
        ends.sort()
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 0.012)
        for (v, name, c, p), y in zip(ends, ys):
            ax.text(306, y, f"{name}  {v:.3f}", va="center", fontsize=9.5, color=INK, weight="semibold" if "Pursuit" in name else "normal")
        ax.set_xlim(55, 300)
        ax.set_xticks([60, 120, 180, 240, 300])
        ax.set_xlabel("training step")
        grid(ax)
        ax.set_title(f"budget {tier.replace('~', '≈')}", color=INK)
    axes[0].set_ylabel("validation loss (lower is better)")
    fig.subplots_adjust(wspace=0.75)
    save(fig, "06_training_curves.png")


# ---- 7. loss vs parameters ---------------------------------------------------------------------------
def fig_loss_params():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    pts = []
    for tier, rows in TIERS.items():
        for r in rows:
            if "ols" not in r["method"]:
                pts.append((r["params"], r["val"], r["method"]))
    for r in R["larger_budgets_lora"]:
        pts.append((r["params"], r["val"], r["method"]))
    pl = sorted((p, v) for p, v, m in pts if "PursuitLoRA" in m)
    ax.plot(*zip(*pl), color=PL, lw=2.2, zorder=3)
    for p, v, m in pts:
        big = "Pursuit" in m
        ax.scatter(p, v, s=110 if big else 65, color=color_of(m), edgecolor=SURF, linewidth=1.5, zorder=4)
    place = {0: (1 / 1.12, 0.014, "right"), 1: (1 / 1.12, -0.018, "right"), 2: (1 / 1.12, -0.016, "right"),
             3: (1.12, -0.012, "left")}
    for i, (p, v) in enumerate(pl):
        fx, dy, ha = place[i]
        ax.text(p * fx, v + dy, f"{v:.3f}", fontsize=10.5, color=PL, ha=ha, va="center", weight="semibold")
    for p, v, m in pts:
        if m.startswith("LoRA r=4") or m == "LoRA r=1 (all matrices)":
            ax.text(p * 1.12, v, m.replace(" (all matrices)", ", all matrices"), fontsize=10, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("validation loss (lower is better)")
    ax.set_title("Loss against parameters: PursuitLoRA (blue line) and every baseline run", color=INK)
    ax.axhline(BASE, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(700, BASE - 0.012, f"base model, no fine-tuning: {BASE:.3f}", color=MUTED, fontsize=10, va="top")
    ax.set_xlim(600, 2.5e6)
    grid(ax, "both")
    fam = [("PursuitLoRA", PL), ("prompt tuning", ORANGE), ("LayerNorm tuning", AQUA), ("LoRA / QLoRA", YELLOW),
           ("(IA)³", MAGENTA), ("VeRA, BitFit", MUTED)]
    h = [Line2D([], [], ls="", marker="o", ms=9, color=c, mec=SURF) for _, c in fam]
    ax.legend(h, [n for n, _ in fam], loc="lower left", fontsize=10.5, ncol=2, handletextpad=0.2, columnspacing=1.2)
    save(fig, "07_loss_vs_parameters.png")


# ---- 8. ranking per budget ---------------------------------------------------------------------------
def fig_rankings():
    tiers = ["~1.5k", "~9k", "~37k", "~55k"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    for ax, tier in zip(axes, tiers):
        rows = sorted([r for r in TIERS[tier] if "ols" not in r["method"]], key=lambda r: r["val"])
        if tier == "~55k":
            rows = sorted(rows + [r for r in TIERS["~83k"]], key=lambda r: r["val"])
        names = [r["method"].replace("PursuitLoRA (sigma)", "PursuitLoRA").replace("(IA)3", "(IA)³")
                 .replace("(all biases)", "(all biases,\n82,944 params)") for r in rows]
        y = np.arange(len(rows))[::-1]
        for yi, r, n in zip(y, rows, names):
            c = color_of(n)
            ax.hlines(yi, 1.2, r["val"], color=GRID, lw=1)
            ax.scatter(r["val"], yi, s=90, color=c, edgecolor=SURF, lw=1.5, zorder=3)
            ax.text(r["val"] + 0.006, yi + 0.28, f"{r['val']:.3f}", fontsize=9, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace(" (", "\n(", 1) for n in names], fontsize=9)
        for t, n in zip(ax.get_yticklabels(), names):
            if "Pursuit" in n:
                t.set_weight("semibold")
                t.set_color(PL)
        ax.set_xlim(1.2, 1.65)
        ax.set_ylim(-0.6, max(len(rows), 4) - 0.4)
        ax.set_title(f"≈{tier[1:]}", color=INK)
        grid(ax, "x")
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel("validation loss (lower is better)")
    fig.subplots_adjust(wspace=1.05)
    save(fig, "08_rankings.png")


# ---- 9. three seeds ---------------------------------------------------------------------------------
def fig_seeds():
    rows = R["three_seeds"]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.3 * len(rows), 4.4))
    for ax, r in zip(np.atleast_1d(axes), rows):
        a, b = r["PursuitLoRA (sigma)"], r["baseline_vals"]
        for i, (x, y) in enumerate(zip(a, b)):
            ax.plot([0, 1], [x, y], color=GRID, lw=1.2, zorder=1)
        ax.scatter([0] * 3, a, s=80, color=PL, edgecolor=SURF, lw=1.5, zorder=3)
        ax.scatter([1] * 3, b, s=80, color=color_of(r["baseline"]), edgecolor=SURF, lw=1.5, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PursuitLoRA", r["baseline"].replace("(IA)3", "(IA)³").replace(" (", "\n(", 1)], fontsize=10)
        ax.set_xlim(-0.4, 1.4)
        lo, hi = min(a + b), max(a + b)
        pad = max(0.01, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        d = r["mean_diff"]
        ax.set_title(f"≈{r['budget'][1:]}:  mean difference {d:+.3f}".replace("-", "−") + f"\npaired t-test p = {r['p_paired_t']:.2f}",
                     color=INK, fontsize=13)
        grid(ax)
    np.atleast_1d(axes)[0].set_ylabel("validation loss, 3 seeds")
    fig.subplots_adjust(wspace=0.45)
    save(fig, "09_three_seeds.png")


# ---- 10. design choices at 9k -----------------------------------------------------------------------
def fig_ablation():
    rows = sorted(R["ablation_9k"], key=lambda r: r["dev"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    y = np.arange(len(rows))[::-1]
    lo = min(r["dev"] for r in rows) - 0.004
    for yi, r in zip(y, rows):
        ax.hlines(yi, lo, r["dev"], color=GRID, lw=1)
        ax.scatter(r["dev"], yi, s=90, color=PL, edgecolor=SURF, lw=1.5, zorder=3)
        ax.text(r["dev"] + 0.0004, yi + 0.3, f"{r['dev']:.4f}", fontsize=9.5, color=INK2)
    lab = [r["variant"].replace("per parameter", "÷ cost") + ("" if r["params"] == 8452 else f"  ({r['params']:,} params)")
           for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(lo, max(r["dev"] for r in rows) + 0.004)
    ax.set_xlabel("loss on a separate dev split at ≈9k (lower is better)")
    ax.set_title("Design variants, compared on a dev split: the differences are small", color=INK)
    grid(ax, "x")
    save(fig, "10_variants_9k.png")


if __name__ == "__main__":
    fig_concept()
    fig_workflow()
    fig_atom()
    fig_spectrum()
    fig_where()
    fig_curves()
    fig_loss_params()
    fig_rankings()
    fig_seeds()
    fig_ablation()

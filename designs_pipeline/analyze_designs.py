#!/usr/bin/env python3
import argparse
import os
import sys
import csv
from typing import List, Dict

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CUR_DIR)
sys.path.insert(0, ROOT)

from IDH1.analyze_idh1_designs import (
    parse_pdb_atoms,
    build_chain_maps,
    compute_bsa_per_position,
    compute_binder_hla_bsa,
    chain_sequence,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms


def main():
    parser = argparse.ArgumentParser(description="Analyze binder designs BSA with heatmaps and CSVs")
    parser.add_argument("--designs_dir", required=True)
    parser.add_argument("--peptide_seq", required=True)
    parser.add_argument("--specific_positions", default="", help="Comma-separated 1-based positions used for specificity ranking")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    peptide_seq = args.peptide_seq.strip()
    L = len(peptide_seq)
    spec_positions0 = []
    if args.specific_positions.strip():
        spec_positions0 = [int(p.strip())-1 for p in args.specific_positions.split(',') if p.strip()]

    pdb_files = [os.path.join(args.designs_dir, f) for f in os.listdir(args.designs_dir) if f.lower().endswith('.pdb')]
    pdb_files.sort()

    design_to_counts: Dict[str, List[float]] = {}
    design_to_hla_bsa: Dict[str, float] = {}

    def find_peptide_chain_by_seq(chains: Dict[str, Dict[int, str]], peptide_seq: str, min_ident: int = 5):
        best = None
        best_ident = -1
        Lref = len(peptide_seq)
        for ch, resmap in chains.items():
            resis, seq = chain_sequence(resmap)
            for start in range(0, max(1, len(seq) - Lref + 1)):
                window = seq[start:start + Lref]
                ident = sum(1 for a, b in zip(window, peptide_seq) if a == b)
                if ident > best_ident:
                    best_ident = ident
                    best = (ch, resis[start:start + Lref], window)
        if best_ident >= min_ident:
            return best
        return None

    for pdb_path in pdb_files:
        atoms = parse_pdb_atoms(pdb_path)
        chains, chain_atom_coords = build_chain_maps(atoms)
        pep = find_peptide_chain_by_seq(chains, peptide_seq)
        if not pep:
            continue
        peptide_chain, peptide_resis, window_seq = pep
        # Compute per-position BSA
        # Choose binder chain by contacts heuristic from imported module
        best_chain = None
        best_contacts = -1
        # rough heuristic
        for ch, coords in chain_atom_coords.items():
            if ch == peptide_chain:
                continue
            ch_len = len(chains[ch])
            if ch_len < 30 or ch_len > 300:
                continue
            # reuse distance-based contacts for binder selection
            # minimal stub: count coords
            tot = len(coords)
            if tot > best_contacts:
                best_contacts = tot
                best_chain = ch
        if best_chain is None:
            continue

        bsa = compute_bsa_per_position(pdb_path, peptide_chain=peptide_chain, peptide_resis=peptide_resis, binder_chain=best_chain)
        design_name = os.path.basename(pdb_path)
        design_to_counts[design_name] = bsa
        design_to_hla_bsa[design_name] = compute_binder_hla_bsa(pdb_path, peptide_chain=peptide_chain, peptide_resis=peptide_resis, binder_chain=best_chain)

    # Write per-position and summary
    per_csv = os.path.join(args.output_dir, "per_position_contacts.csv")
    with open(per_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["design"] + [f"P{i+1}_{peptide_seq[i]}" for i in range(L)] + ["BSA_binder_vs_HLA"])
        for d, counts in design_to_counts.items():
            w.writerow([d] + counts + [design_to_hla_bsa.get(d, 0.0)])

    def specificity_score(counts: List[float]) -> float:
        if not spec_positions0:
            return sum(counts)
        return float(sum(counts[p] for p in spec_positions0))

    ranked = sorted(design_to_counts.items(), key=lambda kv: (-specificity_score(kv[1]), -sum(kv[1])))
    ranked_csv = os.path.join(args.output_dir, "ranked_designs.csv")
    with open(ranked_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["design", "specificity_score", "total_BSA", "binder_HLA_BSA"])
        for d, counts in ranked:
            w.writerow([d, specificity_score(counts), sum(counts), design_to_hla_bsa.get(d, 0.0)])

    summary_csv = os.path.join(args.output_dir, "summary_table.csv")
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["design", "total_peptide_BSA", "binder_HLA_BSA", "ratio_total_over_binderHLA", "defined_positions_BSA", "ratio_defined_over_binderHLA"])
        for d, counts in ranked:
            total_bsa = float(sum(counts))
            hla = float(design_to_hla_bsa.get(d, 0.0))
            defined = float(sum(counts[p] for p in spec_positions0)) if spec_positions0 else total_bsa
            w.writerow([d, total_bsa, hla, (total_bsa/hla if hla>0 else float('inf')), defined, (defined/hla if hla>0 else float('inf'))])

    # Heatmap rendering helper
    def render_heatmap(ordered_designs: List[str], out_path: str, title: str):
        H_pos = np.array([design_to_counts[d] for d in ordered_designs])
        H_hla = np.array([[design_to_hla_bsa.get(d, 0.0)] for d in ordered_designs])
        fig_h = max(6, 0.18 * len(ordered_designs) + 2)
        max_name_len = max((len(n) for n in ordered_designs), default=20)
        fig_w = max(10, 0.7 * L + 2.5 + min(12, 0.09 * max_name_len))
        fig = plt.figure(figsize=(fig_w, fig_h))
        name_col_width = max(2.0, min(6.0, 0.09 * max_name_len))
        gs = gridspec.GridSpec(1, 3, width_ratios=[name_col_width, L, 1], wspace=0.05)
        axNames = fig.add_subplot(gs[0, 0])
        axL = fig.add_subplot(gs[0, 1])
        axR = fig.add_subplot(gs[0, 2], sharey=axL)

        imL = axL.imshow(H_pos, aspect='auto', cmap='viridis', origin='upper')
        cbarL = fig.colorbar(imL, ax=axL, fraction=0.03, pad=0.01)
        cbarL.set_label('')
        imR = axR.imshow(H_hla, aspect='auto', cmap='viridis', origin='upper')
        cbarR = fig.colorbar(imR, ax=axR, fraction=0.03, pad=0.01)
        cbarR.set_label('binder–HLA BSA (Å²)')

        axL.set_yticks([]); axR.set_yticks([])
        axNames.set_xlim(0, 1); axNames.set_ylim(axL.get_ylim())
        name_tx = mtransforms.blended_transform_factory(axNames.transAxes, axNames.transData)
        for i, name in enumerate(ordered_designs):
            axNames.text(0.98, i, name, ha='right', va='center', fontsize=8, transform=name_tx)
        for s in ('top','right','bottom','left'):
            axNames.spines[s].set_visible(False)
        axNames.set_xticks([]); axNames.set_yticks([])

        axL.set_xticks(range(L))
        axL.set_xticklabels([f"P{i+1}\n{peptide_seq[i]}" for i in range(L)])
        axR.set_xticks([0]); axR.set_xticklabels(["binder–HLA\nBSA"])
        axL.set_xlabel('Peptide position'); axL.set_ylabel('')
        fig.suptitle(title)
        for p in spec_positions0:
            axL.axvline(p-0.5, color='white', linewidth=0.8, alpha=0.6)
            axL.axvline(p+0.5, color='white', linewidth=0.8, alpha=0.6)
        fig.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.06, wspace=0.15)
        fig.savefig(out_path, dpi=200); plt.close(fig)

    # Variants
    ordered_specific = [d for d, _ in ranked]
    render_heatmap(ordered_specific, os.path.join(args.output_dir, "contacts_heatmap.png"), "Binder–peptide BSA and binder–HLA BSA")
    ordered_total = sorted(design_to_counts.keys(), key=lambda d: -float(sum(design_to_counts[d])))
    render_heatmap(ordered_total, os.path.join(args.output_dir, "contacts_heatmap_by_total.png"), "Ranked by total peptide BSA")
    def r_total(d):
        h = float(design_to_hla_bsa.get(d, 0.0)); t = float(sum(design_to_counts[d])); return (t/h) if h>0 else float('inf')
    ordered_ratio_total = sorted(design_to_counts.keys(), key=lambda d: -r_total(d))
    render_heatmap(ordered_ratio_total, os.path.join(args.output_dir, "contacts_heatmap_by_ratio_total_over_hla.png"), "Ranked by total/HLA ratio")
    def r_defined(d):
        h = float(design_to_hla_bsa.get(d, 0.0)); defined = float(sum(design_to_counts[d][p] for p in spec_positions0)) if spec_positions0 else float(sum(design_to_counts[d])); return (defined/h) if h>0 else float('inf')
    ordered_ratio_defined = sorted(design_to_counts.keys(), key=lambda d: -r_defined(d))
    render_heatmap(ordered_ratio_defined, os.path.join(args.output_dir, "contacts_heatmap_by_ratio_defined_over_hla.png"), "Ranked by defined/HLA ratio")


if __name__ == "__main__":
    main()



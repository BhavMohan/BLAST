#!/usr/bin/env python3
import os, sys, io, tempfile, zipfile, shutil
from typing import List, Dict
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CUR_DIR)
sys.path.insert(0, ROOT)

from IDH1.analyze_idh1_designs import (
    parse_pdb_atoms,
    build_chain_maps,
    chain_sequence,
    compute_bsa_per_position,
    compute_binder_hla_bsa,
)

st.set_page_config(page_title="Designs BSA (standalone)", layout="wide")
st.title("Designs BSA (standalone)")

peptide_seq = st.text_input("Peptide sequence", value="KPIIIGHHAY")
spec_positions = st.text_input("Specific residues (1-based, comma)", value="1,7,9")
folder = st.text_input("Designs folder (optional if uploading)", value="")
uploads = st.file_uploader("Drag & drop PDBs or a ZIP", type=["pdb","zip"], accept_multiple_files=True)
ranking_choice = st.selectbox(
    "Ranking method",
    [
        "Defined positions only",
        "Total peptide",
        "Defined/HLA ratio",
        "Total/HLA ratio",
    ],
    index=0,
)
run = st.button("Run")

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

if run:
    tmp_dir = None
    run_dir = folder
    if uploads:
        tmp_dir = tempfile.mkdtemp(prefix="designs_upload_")
        for uf in uploads:
            name = uf.name; data = uf.read()
            if name.lower().endswith('.zip'):
                zp = os.path.join(tmp_dir, name)
                with open(zp,'wb') as f: f.write(data)
                with zipfile.ZipFile(zp) as zf: zf.extractall(tmp_dir)
                os.remove(zp)
            else:
                with open(os.path.join(tmp_dir, name),'wb') as f: f.write(data)
        run_dir = tmp_dir

    if not run_dir or not os.path.isdir(run_dir):
        st.error("Provide a folder or upload PDBs/ZIP.")
    else:
        files = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.lower().endswith('.pdb')]
        if not files:
            st.error("No PDBs found.")
        else:
            spec0 = [int(p.strip())-1 for p in spec_positions.split(',') if p.strip()]
            design_to_counts: Dict[str, List[float]] = {}
            design_to_hla: Dict[str, float] = {}
            for fp in files:
                atoms = parse_pdb_atoms(fp)
                chains, chain_atoms = build_chain_maps(atoms)
                pep = find_peptide_chain_by_seq(chains, peptide_seq)
                if not pep:
                    continue
                pch, pres, _ = pep
                # Pick a binder chain simply by largest atom list other than peptide
                other = [ (ch,len(chain_atoms[ch])) for ch in chains if ch!=pch ]
                if not other:
                    continue
                best_chain = sorted(other, key=lambda x: -x[1])[0][0]
                bsa = compute_bsa_per_position(fp, pch, pres, best_chain)
                design_to_counts[os.path.basename(fp)] = bsa
                design_to_hla[os.path.basename(fp)] = compute_binder_hla_bsa(fp, pch, pres, best_chain)

            if not design_to_counts:
                st.error("Could not map peptide in uploaded designs.")
            else:
                def spec_score(d):
                    return float(sum(design_to_counts[d][p] for p in spec0)) if spec0 else float(sum(design_to_counts[d]))
                # Determine ordering from dropdown
                if ranking_choice == "Defined positions only":
                    ranked = sorted(design_to_counts.keys(), key=lambda d: (-spec_score(d), -float(sum(design_to_counts[d]))))
                elif ranking_choice == "Total peptide":
                    ranked = sorted(design_to_counts.keys(), key=lambda d: -float(sum(design_to_counts[d])))
                elif ranking_choice == "Defined/HLA ratio":
                    ranked = sorted(
                        design_to_counts.keys(),
                        key=lambda d: -(
                            ((sum(design_to_counts[d][p] for p in spec0) if spec0 else sum(design_to_counts[d])) /
                             (design_to_hla.get(d,0.0) or float('inf')))
                        ),
                    )
                else:  # Total/HLA ratio
                    ranked = sorted(
                        design_to_counts.keys(),
                        key=lambda d: -(
                            (sum(design_to_counts[d]) / (design_to_hla.get(d,0.0) or float('inf')))
                        ),
                    )
                # Build matrices
                H_pos = np.array([design_to_counts[d] for d in ranked])
                H_hla = np.array([[design_to_hla.get(d,0.0)] for d in ranked])
                H_total_ratio = np.array([[ (sum(design_to_counts[d])/(design_to_hla.get(d,0.0) or np.inf)) ] for d in ranked])
                H_defined_ratio = np.array([[ ( (sum(design_to_counts[d][p] for p in spec0) if spec0 else sum(design_to_counts[d])) / (design_to_hla.get(d,0.0) or np.inf) ) ] for d in ranked])

                fig_h = max(6, 0.18 * len(ranked) + 2)
                fig_w = max(12, 0.7 * len(peptide_seq) + 6)
                fig = plt.figure(figsize=(fig_w, fig_h))
                name_col = max(2.0, min(6.0, 0.09 * max((len(n) for n in ranked), default=20)))
                gs = gridspec.GridSpec(1, 5, width_ratios=[name_col, len(peptide_seq), 1, 1, 1], wspace=0.05)
                axNames = fig.add_subplot(gs[0,0])
                axPos = fig.add_subplot(gs[0,1])
                axHLA = fig.add_subplot(gs[0,2], sharey=axPos)
                axR1 = fig.add_subplot(gs[0,3], sharey=axPos)
                axR2 = fig.add_subplot(gs[0,4], sharey=axPos)

                imPos = axPos.imshow(H_pos, aspect='auto', cmap='viridis', origin='upper')
                fig.colorbar(imPos, ax=axPos, fraction=0.03, pad=0.01)
                imHLA = axHLA.imshow(H_hla, aspect='auto', cmap='viridis', origin='upper')
                cHLA = fig.colorbar(imHLA, ax=axHLA, fraction=0.03, pad=0.01)
                cHLA.set_label('binder–HLA BSA (Å²)')
                imR1 = axR1.imshow(H_total_ratio, aspect='auto', cmap='viridis', origin='upper')
                cR1 = fig.colorbar(imR1, ax=axR1, fraction=0.03, pad=0.01)
                cR1.set_label('Total/HLA')
                imR2 = axR2.imshow(H_defined_ratio, aspect='auto', cmap='viridis', origin='upper')
                cR2 = fig.colorbar(imR2, ax=axR2, fraction=0.03, pad=0.01)
                cR2.set_label('Defined/HLA')

                # Names
                axPos.set_yticks([]); axHLA.set_yticks([]); axR1.set_yticks([]); axR2.set_yticks([])
                axNames.set_xlim(0,1); axNames.set_ylim(axPos.get_ylim())
                tx = mtransforms.blended_transform_factory(axNames.transAxes, axNames.transData)
                for i,name in enumerate(ranked):
                    axNames.text(0.98, i, name, ha='right', va='center', fontsize=8, transform=tx)
                for s in ('top','right','bottom','left'):
                    axNames.spines[s].set_visible(False)
                axNames.set_xticks([]); axNames.set_yticks([])

                axPos.set_xticks(range(len(peptide_seq)))
                axPos.set_xticklabels([f"P{i+1}\n{peptide_seq[i]}" for i in range(len(peptide_seq))])
                axHLA.set_xticks([0]); axHLA.set_xticklabels(["binder–HLA\nBSA"])
                axR1.set_xticks([0]); axR1.set_xticklabels(["Total/HLA"])
                axR2.set_xticks([0]); axR2.set_xticklabels(["Defined/HLA"])

                st.pyplot(fig)
                # Offer downloads for all heatmaps
                import tempfile
                from PIL import Image
                tmpdir = tempfile.mkdtemp(prefix='designs_dl_')
                paths = {
                    'contacts_heatmap.png': os.path.join(tmpdir,'contacts_heatmap.png'),
                    'contacts_heatmap_by_total.png': os.path.join(tmpdir,'contacts_heatmap_by_total.png'),
                    'contacts_heatmap_by_ratio_total_over_hla.png': os.path.join(tmpdir,'contacts_heatmap_by_ratio_total_over_hla.png'),
                    'contacts_heatmap_by_ratio_defined_over_hla.png': os.path.join(tmpdir,'contacts_heatmap_by_ratio_defined_over_hla.png'),
                }
                # Render and save each ordering
                orderings = {
                    'contacts_heatmap.png': ranked,
                    'contacts_heatmap_by_total.png': sorted(design_to_counts.keys(), key=lambda d: -float(sum(design_to_counts[d]))),
                    'contacts_heatmap_by_ratio_total_over_hla.png': sorted(design_to_counts.keys(), key=lambda d: -( (sum(design_to_counts[d])/(design_to_hla.get(d,0.0) or np.inf)) )),
                    'contacts_heatmap_by_ratio_defined_over_hla.png': sorted(design_to_counts.keys(), key=lambda d: -( ((sum(design_to_counts[d][p] for p in spec0) if spec0 else sum(design_to_counts[d]))/(design_to_hla.get(d,0.0) or np.inf)) )),
                }
                for name, order in orderings.items():
                    Hpos = np.array([design_to_counts[d] for d in order]); Hhla = np.array([[design_to_hla.get(d,0.0)] for d in order])
                    fig2 = plt.figure(figsize=(fig_w, fig_h))
                    gs2 = gridspec.GridSpec(1, 5, width_ratios=[name_col, len(peptide_seq), 1, 1, 1], wspace=0.05)
                    axN2 = fig2.add_subplot(gs2[0,0]); axP2 = fig2.add_subplot(gs2[0,1]); axH2 = fig2.add_subplot(gs2[0,2]); axT2 = fig2.add_subplot(gs2[0,3]); axD2 = fig2.add_subplot(gs2[0,4])
                    axP2.imshow(Hpos, aspect='auto', cmap='viridis', origin='upper')
                    axH2.imshow(Hhla, aspect='auto', cmap='viridis', origin='upper')
                    tr = np.array([[ (sum(design_to_counts[d])/(design_to_hla.get(d,0.0) or np.inf)) ] for d in order])
                    dr = np.array([[ ((sum(design_to_counts[d][p] for p in spec0) if spec0 else sum(design_to_counts[d]))/(design_to_hla.get(d,0.0) or np.inf)) ] for d in order])
                    axT2.imshow(tr, aspect='auto', cmap='viridis', origin='upper')
                    axD2.imshow(dr, aspect='auto', cmap='viridis', origin='upper')
                    axP2.set_yticks([]); axH2.set_yticks([]); axT2.set_yticks([]); axD2.set_yticks([])
                    axN2.set_xlim(0,1); axN2.set_ylim(axP2.get_ylim()); tx2 = mtransforms.blended_transform_factory(axN2.transAxes, axN2.transData)
                    for i,nm in enumerate(order): axN2.text(0.98, i, nm, ha='right', va='center', fontsize=8, transform=tx2)
                    axN2.set_xticks([]); axN2.set_yticks([])
                    fig2.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.06, wspace=0.15)
                    fig2.savefig(paths[name], dpi=200); plt.close(fig2)
                # Zip and provide download
                import zipfile
                zip_path = os.path.join(tmpdir, 'all_heatmaps.zip')
                with zipfile.ZipFile(zip_path, 'w') as z:
                    for n,pth in paths.items():
                        if os.path.exists(pth): z.write(pth, arcname=n)
                with open(zip_path, 'rb') as f:
                    st.download_button("Download all heatmaps (zip)", f, file_name="all_heatmaps.zip")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)



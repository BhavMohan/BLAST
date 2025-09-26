## Binder designs BSA analysis pipeline

Given a peptide sequence, specificity positions, and a folder of PDB binder designs bound to the peptide–HLA complex, this pipeline computes per-position buried surface area (BSA), binder–HLA interface BSA (with peptide removed), and renders four heatmaps plus CSV summaries.

### Usage

```bash
python analyze_designs.py \
  --designs_dir "/path/to/designs_pdbs" \
  --peptide_seq KPIIIGHHAY \
  --specific_positions "1,7,9" \
  --output_dir "/path/to/output"
```

Outputs written under `--output_dir`:
- `per_position_contacts.csv` — per-position peptide BSA and binder–HLA BSA per design
- `ranked_designs.csv` — ranking by specificity positions BSA
- `summary_table.csv` — totals and ratios per design
- Heatmaps:
  - `contacts_heatmap.png` (ranked by specificity positions)
  - `contacts_heatmap_by_total.png`
  - `contacts_heatmap_by_ratio_total_over_hla.png`
  - `contacts_heatmap_by_ratio_defined_over_hla.png`



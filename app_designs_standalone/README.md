## Designs BSA (standalone Streamlit app)

Inputs:
- Peptide sequence
- Specific residues (1-based, comma)
- Folder of PDB designs (or drag & drop)

The app computes per-position peptide BSA, binder–HLA BSA (peptide removed), and renders four heatmaps, plus two extra columns beside the binder–HLA column showing Total/HLA and Defined/HLA ratios.

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit
streamlit run app_designs_standalone/app.py
```



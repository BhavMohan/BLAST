## Standalone BLAST specificity app

Single-file Streamlit app. Provide:
- Peptide sequence
- Inward-facing (avoid) positions (1-based; comma; -1 for last)
- Upload BLAST "FASTA (aligned clusters)" text

Outputs displayed and downloadable:
- `blast_alignment.png`
- `blast_specificity.txt`

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r standalone_blast_app/requirements.txt
streamlit run standalone_blast_app/app.py
```



#!/usr/bin/env python3
"""
Analyze NCBI BLAST "FASTA (aligned clusters)" output to:
- Filter Homo sapiens hits
- Align each hit to a reference peptide (default: ALHGGWTTK)
- For short aligned fragments, fetch ±3 residue flanks from NCBI
- Produce a comparison figure showing matches vs mismatches aligned to the reference

Usage:
  python analyze_blast.py \
      --input "/Users/bmohan/Documents/cursor_binding/Protein Sequence Seqdump.txt" \
      --output "/Users/bmohan/Documents/cursor_binding/blast_alignment.png" \
      --reference ALHGGWTTK \
      --flank_threshold 5 \
      --email your_email@example.com

Notes:
- An email is recommended for NCBI E-utilities compliance.
- A small on-disk cache is used to avoid repeated efetch calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import requests


NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_CACHE_FILE = "ncbi_cache_proteins.json"


@dataclass
class BlastHit:
    header: str
    accession: Optional[str]
    start_pos: Optional[int]  # 1-based inclusive
    end_pos: Optional[int]    # 1-based inclusive
    organism: Optional[str]
    description: Optional[str]
    fragment_seq: str


def load_ncbi_cache(cache_path: str) -> Dict[str, str]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_ncbi_cache(cache_path: str, cache: Dict[str, str]) -> None:
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp_path, cache_path)


def parse_header(header: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str], Optional[str]]:
    """Parse a FASTA cluster header line like:
    >NP_065825.1:268-272 E3 ubiquitin-protein ligase MIB1 [Homo sapiens]
    >7L1C_C:1-9 Chain C, mutant PIK3CA peptide [Homo sapiens]
    Returns (accession, start, end, organism, description)
    """
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]

    # Extract organism inside brackets at the end
    organism_match = re.search(r"\[(.*?)\]\s*$", header)
    organism = organism_match.group(1) if organism_match else None

    # Remove organism for description parsing
    core = header
    if organism_match:
        core = header[: organism_match.start()].strip()

    # Expect leading token like ACCESSION:START-END followed by description
    # ACCESSION can be alnum, underscores, dots, and possibly chain info like 7L1C_C
    pos_match = re.match(r"^(\S+?)(?::(\d+)-(\d+))?(?:\s+(.*))?$", core)
    accession = None
    start = None
    end = None
    description = None
    if pos_match:
        accession = pos_match.group(1)
        if pos_match.group(2) and pos_match.group(3):
            try:
                start = int(pos_match.group(2))
                end = int(pos_match.group(3))
            except ValueError:
                start = None
                end = None
        description = (pos_match.group(4) or "").strip() or None

    return accession, start, end, organism, description


def parse_fasta_clusters(path: str) -> List[BlastHit]:
    hits: List[BlastHit] = []
    current_header: Optional[str] = None
    current_seq_parts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # flush previous
                if current_header is not None:
                    fragment = "".join(current_seq_parts).strip()
                    acc, s, e, org, desc = parse_header(current_header)
                    if fragment:
                        hits.append(
                            BlastHit(
                                header=current_header.strip(),
                                accession=acc,
                                start_pos=s,
                                end_pos=e,
                                organism=org,
                                description=desc,
                                fragment_seq=fragment,
                            )
                        )
                # start new
                current_header = line
                current_seq_parts = []
            else:
                if line:
                    current_seq_parts.append(line.strip())

        # flush last
        if current_header is not None:
            fragment = "".join(current_seq_parts).strip()
            acc, s, e, org, desc = parse_header(current_header)
            if fragment:
                hits.append(
                    BlastHit(
                        header=current_header.strip(),
                        accession=acc,
                        start_pos=s,
                        end_pos=e,
                        organism=org,
                        description=desc,
                        fragment_seq=fragment,
                    )
                )

    return hits


def filter_by_species(hits: List[BlastHit], species: str) -> List[BlastHit]:
    target = (species or "").lower().strip()
    return [h for h in hits if (h.organism or "").lower().strip() == target]


def score_alignment_at(ref: str, frag: str, start_idx: int) -> int:
    """Score alignment placing frag starting at ref[start_idx]. No gaps. +1 match, 0 for 'X' in frag, -1 mismatch."""
    score = 0
    for i, aa in enumerate(frag):
        ref_idx = start_idx + i
        if ref_idx < 0 or ref_idx >= len(ref):
            return -10_000  # invalid placement
        if aa == 'X' or aa == 'x':
            continue
        score += 1 if aa == ref[ref_idx] else -1
    return score


def best_alignment_to_reference(ref: str, frag: str) -> Tuple[int, int]:
    """Return (best_start_index_in_ref, best_score). No gaps, slide across reference."""
    best_start = 0
    best_score = -10_000
    for start in range(0, len(ref) - len(frag) + 1):
        s = score_alignment_at(ref, frag, start)
        if s > best_score:
            best_score = s
            best_start = start
    return best_start, best_score


def fetch_protein_fasta(accession: str, email: Optional[str], api_key: Optional[str], timeout_s: int = 20) -> Optional[str]:
    params = {
        "db": "protein",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
    }
    headers = {}
    if email:
        headers["User-Agent"] = f"blast-analyzer/1.0 ({email})"
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(NCBI_EFETCH_URL, params=params, headers=headers, timeout=timeout_s)
    if resp.status_code != 200:
        return None
    text = resp.text
    if not text.startswith(">"):
        return None
    return text


def fasta_sequence_from_text(fasta_text: str) -> Optional[str]:
    lines = fasta_text.splitlines()
    if not lines or not lines[0].startswith(">"):
        return None
    seq_lines = [ln.strip() for ln in lines[1:] if ln and not ln.startswith(">")]
    return "".join(seq_lines) if seq_lines else None


def get_protein_sequence(accession: str, cache: Dict[str, str], email: Optional[str], api_key: Optional[str]) -> Optional[str]:
    if accession in cache:
        return cache[accession]
    fasta_text = fetch_protein_fasta(accession, email=email, api_key=api_key)
    if fasta_text is None:
        return None
    seq = fasta_sequence_from_text(fasta_text)
    if seq:
        cache[accession] = seq
    return seq


def extract_flank_sequence(full_seq: str, start_1b: int, end_1b: int, flank: int = 3) -> str:
    """Return context string like NNN[MATCH]NNN given 1-based inclusive indices."""
    n = len(full_seq)
    s0 = max(1, start_1b - flank)
    e0 = min(n, end_1b + flank)
    left = full_seq[s0 - 1 : start_1b - 1]
    match = full_seq[start_1b - 1 : end_1b]
    right = full_seq[end_1b : e0]
    return f"{left}[{match}]{right}"


def build_alignment_rows(ref: str, hits: List[BlastHit]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for h in hits:
        start_in_ref, score = best_alignment_to_reference(ref, h.fragment_seq)
        # compute identity stats at best placement
        matches = 0
        mismatches = 0
        effective = 0
        for i, aa in enumerate(h.fragment_seq):
            ri = start_in_ref + i
            if 0 <= ri < len(ref):
                if aa.upper() == 'X':
                    continue
                effective += 1
                if aa.upper() == ref[ri].upper():
                    matches += 1
                else:
                    mismatches += 1
        identity = (matches / effective) if effective > 0 else 0.0
        rows.append({
            "hit": h,
            "ref_start": start_in_ref,
            "score": score,
            "matches": matches,
            "mismatches": mismatches,
            "identity": identity,
        })
    # Initial ordering by similarity (score desc, identity desc, longer first)
    rows.sort(key=lambda r: (-int(r["score"]), -float(r["identity"]), -len(r["hit"].fragment_seq)))  # type: ignore
    return rows


def normalize_base_name(description: Optional[str]) -> str:
    if not description:
        return ""
    name = description
    # Drop leading Chain labels
    name = re.sub(r"^Chain\s+[A-Z],\s*", "", name, flags=re.IGNORECASE)
    # Remove isoform qualifiers
    name = re.sub(r"\s*,?\s*isoform\s+[^,\[]+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*,?\s*isoform\s*x?\d+", "", name, flags=re.IGNORECASE)
    # Remove common qualifiers
    name = re.sub(r"\b(precursor|partial|variant)\b", "", name, flags=re.IGNORECASE)
    # Collapse whitespace and punctuation spacing
    name = re.sub(r"\s+", " ", name).strip(" ,-")
    return name.strip()


def group_isoforms(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        h: BlastHit = r["hit"]  # type: ignore
        base = normalize_base_name(h.description) or (h.accession or "")
        groups.setdefault(base, []).append(r)

    representatives: List[Dict[str, object]] = []
    for base, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda r: (-int(r["score"]), -float(r.get("identity", 0.0)), -len(r["hit"].fragment_seq)))  # type: ignore
        rep = dict(lst_sorted[0])
        rep["group_base_name"] = base
        rep["group_count"] = len(lst)
        rep["group_members"] = lst
        representatives.append(rep)

    representatives.sort(key=lambda r: (-int(r["score"]), -float(r.get("identity", 0.0)), -len(r["hit"].fragment_seq)))  # type: ignore
    return representatives


def plot_alignment(ref: str,
                   rows: List[Dict[str, object]],
                   out_path: str,
                   context_map: Dict[str, str],
                   chosen_positions: Optional[List[int]] = None,
                   selection_order: Optional[Dict[int, int]] = None) -> None:
    # rows are expected to be grouped representatives
    num_rows = len(rows)
    ref_len = len(ref)
    # Prepare row labels
    label_lefts: List[str] = [f"Reference — {ref}"]
    for row in rows:
        h: BlastHit = row["hit"]  # type: ignore
        base = row.get("group_base_name") or (h.description or "")
        label = base if base else (h.accession or "(no accession)")
        cnt = int(row.get("group_count", 1))
        if cnt > 1:
            label += f"  (+{cnt-1} isoform(s))"
        label_lefts.append(label)

    max_label_len = max((len(s) for s in label_lefts), default=12)
    # Figure size heuristic: width includes space for reference, labels, and context
    total_rows = num_rows + 1  # include reference row
    fig_height = max(3, 0.3 * total_rows + 1.8)
    width_for_ref = max(6.0, ref_len * 0.6)
    width_for_labels = min(12.0, 0.12 * max_label_len + 0.6)
    width_for_context = 4.0
    fig_width = width_for_ref + width_for_labels + width_for_context
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax.set_xlim(-0.5, ref_len + 12)
    ax.set_ylim(-0.5, total_rows - 0.5)
    ax.set_xticks(range(ref_len))
    ax.set_xticklabels([f"{i+1}\n{aa}" for i, aa in enumerate(ref)])
    # Use ytick labels for left-side text to avoid clipping
    y_positions = [total_rows - idx - 1 for idx in range(total_rows)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(label_lefts, fontsize=8)
    ax.tick_params(axis='y', length=0)
    ax.set_ylabel("Hits")
    ax.set_xlabel("Reference position (ALHGGWTTK by default)")
    ax.set_title("Alignment to reference peptide")

    # Draw reference row
    y_ref = total_rows - 1
    for x, aa in enumerate(ref):
        rect = plt.Rectangle((x - 0.45, y_ref - 0.4), 0.9, 0.8, facecolor="#cfe8ff", edgecolor="#6ba6ff")
        ax.add_patch(rect)
        ax.text(x, y_ref, aa, ha="center", va="center", fontsize=9, color="#1f3b70")

    # Draw hit rows
    for idx, row in enumerate(rows):
        h: BlastHit = row["hit"]  # type: ignore
        start = int(row["ref_start"])  # 0-based
        frag = h.fragment_seq
        y = total_rows - (idx + 2)  # one row below reference

        # Determine which selected positions to tint for this row (ALL selected mismatches or undefined positions)
        tint_positions: set = set()
        if chosen_positions:
            # Skip rows identical to reference (full-length, same seq, aligned at start)
            if not (len(frag) == len(ref) and frag.upper() == ref.upper() and start == 0):
                for p in chosen_positions:
                    idx_in_frag = p - start
                    if 0 <= idx_in_frag < len(frag):
                        aa_p = frag[idx_in_frag]
                        if aa_p.upper() != ref[p].upper() or aa_p.upper() == 'X':
                            tint_positions.add(p)
                    else:
                        # No aligned residue at this selected position -> tint
                        tint_positions.add(p)

        # Draw boxes for all reference positions; color based on alignment state
        for x in range(ref_len):
            idx_in_frag = x - start
            aa = frag[idx_in_frag] if 0 <= idx_in_frag < len(frag) else None
            if aa is None:
                # no alignment at this position -> red
                color = "#f9c0c0"
            else:
                if aa.upper() == 'X':
                    color = "#dddddd"  # unknown/ambiguous residue
                else:
                    is_match = aa.upper() == ref[x].upper()
                    color = "#c6e48b" if is_match else "#f9c0c0"
            highlight = chosen_positions is not None and x in set(chosen_positions)
            edge = "#000000" if highlight else "#aaaaaa"
            lw = 1.8 if highlight else 1.0
            rect = plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, facecolor=color, edgecolor=edge, linewidth=lw)
            ax.add_patch(rect)
            # Add gray tint overlay for all selected discriminative mismatches in this row
            if x in tint_positions:
                overlay = plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, facecolor="#9e9e9e", edgecolor=None, alpha=0.35)
                ax.add_patch(overlay)
            if aa is not None:
                ax.text(x, y, aa, ha="center", va="center", fontsize=9, color="#222222")

    # Annotate selection order below reference row if provided
    if chosen_positions and selection_order:
        for x in chosen_positions:
            order = selection_order.get(x)
            if order is not None:
                ax.text(x, y_ref - 0.75, str(order), ha="center", va="top", fontsize=8, color="#000000")

        # Right-side context (if present)
        ctx = context_map.get(h.header)
        if ctx:
            ax.text(ref_len + 0.5, y, ctx, ha="left", va="center", fontsize=8, family="monospace")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BLAST aligned clusters vs reference peptide")
    parser.add_argument("--input", required=True, help="Path to FASTA (aligned clusters) file")
    parser.add_argument("--output", required=True, help="Path to output figure (PNG/PDF)")
    parser.add_argument("--reference", default="ALHGGWTTK", help="Reference peptide sequence")
    parser.add_argument("--flank_threshold", type=int, default=5, help="If fragment length <= threshold, fetch ±3 residue context")
    parser.add_argument("--report", default=None, help="Optional path to write specificity report (txt)")
    parser.add_argument("--specificity_mode", default="conservative", choices=["conservative", "practical"], help="Treatment of unknown residues '?' when evaluating specificity")
    parser.add_argument("--email", default=None, help="Your email for NCBI requests (recommended)")
    parser.add_argument("--ncbi_api_key", default=None, help="NCBI API key (optional)")
    parser.add_argument("--nocache", action="store_true", help="Ignore on-disk cache and refetch from NCBI")
    parser.add_argument("--species", default="Homo sapiens", help="Species to include (exact match to FASTA header organism)")
    parser.add_argument("--avoid_positions", default="2,-1", help="Comma-separated P positions to avoid first; use -1 for last")
    parser.add_argument("--highlight_mode", default="window", choices=["window", "greedy"], help="Which positions to highlight in the figure: minimal unique window or greedy set")
    parser.add_argument("--negatives_input", default=None, help="Optional path to a FASTA (aligned clusters) file of additional hits to exclude (e.g., results from X-pattern BLAST)")
    parser.add_argument("--negatives_species", default="all", help="Species filter for negatives; 'all' to include all")
    parser.add_argument("--matrix_output", default=None, help="Optional path to write substitution specificity matrix figure (PNG/PDF)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print("Parsing clusters…", file=sys.stderr)
    hits = parse_fasta_clusters(input_path)
    hs_hits = filter_by_species(hits, args.species)
    # Remove hits that contain ambiguous amino acids 'X'
    hs_hits = [h for h in hs_hits if 'X' not in h.fragment_seq.upper()]
    if not hs_hits:
        print(f"No {args.species} hits found.", file=sys.stderr)
        sys.exit(2)
    print(f"Found {len(hs_hits)} {args.species} hits.", file=sys.stderr)

    print("Computing alignments to reference…", file=sys.stderr)
    rows = build_alignment_rows(args.reference, hs_hits)

    # Prepare context strings and fetch full protein sequences when possible
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), NCBI_CACHE_FILE)
    cache: Dict[str, str] = {} if args.nocache else load_ncbi_cache(cache_path)
    context_map: Dict[str, str] = {}
    full_seq_map: Dict[str, Optional[str]] = {}

    print("Fetching protein sequences (as available) and context…", file=sys.stderr)
    for row in rows:
        h: BlastHit = row["hit"]  # type: ignore
        if h.accession and (h.start_pos and h.end_pos):
            try:
                if h.accession not in full_seq_map:
                    full_seq = get_protein_sequence(h.accession, cache, email=args.email, api_key=args.ncbi_api_key)
                    full_seq_map[h.accession] = full_seq
                    # Be polite to NCBI
                    time.sleep(0.34)
                else:
                    full_seq = full_seq_map[h.accession]
                if full_seq and len(h.fragment_seq) <= args.flank_threshold:
                    ctx = extract_flank_sequence(full_seq, h.start_pos, h.end_pos, flank=3)
                    context_map[h.header] = ctx
            except Exception:
                context_map[h.header] = ""

    if not args.nocache:
        try:
            save_ncbi_cache(cache_path, cache)
        except Exception:
            pass

    # Defer rendering until after specificity analysis (to annotate selection)
    
    # Build per-hit aligned windows across full reference length using fetched sequences when possible
    def build_mapped_window(ref: str, row: Dict[str, object]) -> str:
        h: BlastHit = row["hit"]  # type: ignore
        start_in_ref: int = row["ref_start"]  # 0-based
        L = len(ref)
        window = ["?"] * L
        # If we have a full protein sequence and coordinates, map the entire reference span
        full_seq = None
        if h.accession and (h.start_pos and h.end_pos):
            full_seq = full_seq_map.get(h.accession)
        if full_seq:
            # protein position corresponding to ref index j is p = start_pos + (j - start_in_ref)
            for j in range(L):
                p = (h.start_pos or 0) + (j - start_in_ref)
                if p >= 1 and p <= len(full_seq):
                    window[j] = full_seq[p - 1]
        else:
            # fallback: only known fragment aligned region
            frag = h.fragment_seq
            for i, aa in enumerate(frag):
                j = start_in_ref + i
                if 0 <= j < L:
                    window[j] = aa
        return "".join(window)

    mapped_windows: List[str] = [build_mapped_window(args.reference, r) for r in rows]

    # Optionally read negatives file and incorporate as additional windows to exclude
    neg_rows: List[Dict[str, object]] = []
    neg_windows: List[str] = []
    if args.negatives_input and os.path.exists(args.negatives_input):
        print(f"Parsing negatives from {args.negatives_input} …", file=sys.stderr)
        neg_hits = parse_fasta_clusters(args.negatives_input)
        if (args.negatives_species or "all").lower() != "all":
            neg_hits = filter_by_species(neg_hits, args.negatives_species)
        # do not remove X from negatives; treat conservatively later
        neg_rows = build_alignment_rows(args.reference, neg_hits)
        neg_windows = [build_mapped_window(args.reference, r) for r in neg_rows]

    # Exclude hits identical to the reference peptide (same length, same sequence, aligned at start)
    considered_rows: List[Dict[str, object]] = []
    windows_considered: List[str] = []
    identical_count = 0
    for r, win in zip(rows, mapped_windows):
        h: BlastHit = r["hit"]  # type: ignore
        if len(h.fragment_seq) == len(args.reference) and h.fragment_seq.upper() == args.reference.upper() and int(r["ref_start"]) == 0:
            identical_count += 1
            continue
        considered_rows.append(r)
        windows_considered.append(win)

    # Add negatives to consideration, ignoring any sequences identical to reference at start
    for r, win in zip(neg_rows, neg_windows):
        h: BlastHit = r["hit"]  # type: ignore
        if len(h.fragment_seq) == len(args.reference) and h.fragment_seq.upper() == args.reference.upper() and int(r["ref_start"]) == 0:
            continue
        considered_rows.append(r)
        windows_considered.append(win)

    # Specificity analysis
    def discriminative_positions(ref: str, windows: List[str], mode: str) -> Tuple[List[int], List[int]]:
        # returns (chosen_positions 0-based, uncovered_hits)
        L = len(ref)
        N = len(windows)
        remaining = set(range(N))
        chosen: List[int] = []
        # precompute coverage per position
        cover: List[set] = []
        for i in range(L):
            cov = set()
            for h in range(N):
                aa = windows[h][i]
                if (aa == '?' or aa.upper() == 'X') and mode == 'conservative':
                    continue
                if aa != '?' and aa.upper() == ref[i].upper():
                    continue
                # differs (or unknown in practical mode)
                cov.add(h)
            cover.append(cov)
        
        def greedy_pass(allowed: Optional[set]) -> None:
            nonlocal remaining, chosen
            while remaining:
                best_i = None
                best_gain = -1
                for i in range(L):
                    if allowed is not None and i not in allowed:
                        continue
                    gain = len(cover[i] & remaining)
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                if best_gain <= 0 or best_i is None:
                    break
                chosen.append(best_i)
                remaining -= cover[best_i]

        # Prefer avoiding positions provided via CLI (1-based; -1 for last)
        avoid_tokens = [t.strip() for t in (args.avoid_positions or "").split(',') if t.strip()]
        avoid: set = set()
        for tok in avoid_tokens:
            try:
                val = int(tok)
            except ValueError:
                continue
            idx0 = (L - 1) if val == -1 else (val - 1)
            if 0 <= idx0 < L:
                avoid.add(idx0)
        allowed_first = set(range(L)) - avoid
        greedy_pass(allowed_first)
        if remaining:
            greedy_pass(None)
        return chosen, sorted(list(remaining))

    def min_unique_window(ref: str, windows: List[str], mode: str) -> Optional[Tuple[int, int]]:
        L = len(ref)
        N = len(windows)
        for k in range(1, L + 1):
            for s in range(0, L - k + 1):
                ref_sub = ref[s : s + k]
                found_match = False
                for h in range(N):
                    hit_sub = windows[h][s : s + k]
                    if mode == 'conservative':
                        # unknown means potentially match, so require all known and equal; if any '?', treat as match
                        if '?' in hit_sub:
                            found_match = True
                            break
                        if all(a.upper() == b.upper() for a, b in zip(hit_sub, ref_sub)):
                            found_match = True
                            break
                    else:
                        # practical: unknown treated as mismatch; require exact equality with no '?'
                        if '?' in hit_sub:
                            continue
                        if all(a.upper() == b.upper() for a, b in zip(hit_sub, ref_sub)):
                            found_match = True
                            break
                if not found_match:
                    return (s, k)
        return None

    mode = args.specificity_mode
    chosen_cons, uncovered_cons = discriminative_positions(args.reference, windows_considered, 'conservative')
    chosen_prac, uncovered_prac = discriminative_positions(args.reference, windows_considered, 'practical')
    win_cons = min_unique_window(args.reference, windows_considered, 'conservative')
    win_prac = min_unique_window(args.reference, windows_considered, 'practical')

    # Write report
    report_path = args.report or os.path.splitext(args.output)[0] + "_specificity.txt"
    lines: List[str] = []
    lines.append(f"Reference: {args.reference}")
    lines.append(f"{args.species} hits (after removing X): {len(rows)}")
    if identical_count:
        lines.append(f"Identical to reference (ignored for specificity): {identical_count}")
    lines.append("")
    lines.append("Discriminative positions (P = 1-based positions):")
    lines.append(f"  Conservative: {[p+1 for p in chosen_cons]}  (uncovered hits: {len(uncovered_cons)})")
    if chosen_cons:
        aa_cons = ''.join(args.reference[i] for i in chosen_cons)
        pretty_cons = ', '.join(f"P{i+1}:{args.reference[i]}" for i in chosen_cons)
        lines.append(f"    Residues to define: {pretty_cons}  (motif: {aa_cons})")
    lines.append(f"  Practical:    {[p+1 for p in chosen_prac]}  (uncovered hits: {len(uncovered_prac)})")
    if chosen_prac:
        aa_prac = ''.join(args.reference[i] for i in chosen_prac)
        pretty_prac = ', '.join(f"P{i+1}:{args.reference[i]}" for i in chosen_prac)
        lines.append(f"    Residues to define: {pretty_prac}  (motif: {aa_prac})")
    lines.append("")
    lines.append("Minimal contiguous unique window vs hits:")
    lines.append(f"  Conservative: {('none' if win_cons is None else f'P{win_cons[0]+1}..P{win_cons[0]+win_cons[1]} sequence ' + args.reference[win_cons[0]:win_cons[0]+win_cons[1]])}")
    lines.append(f"  Practical:    {('none' if win_prac is None else f'P{win_prac[0]+1}..P{win_prac[0]+win_prac[1]} sequence ' + args.reference[win_prac[0]:win_prac[0]+win_prac[1]])}")
    lines.append("")
    if win_prac is not None:
        s, k = win_prac
        minimal_pattern = ''.join(args.reference[i] if (i >= s and i < s + k) else 'X' for i in range(len(args.reference)))
        lines.append("Recommended specificity using minimal window (practical):")
        lines.append(f"  Positions: P{s+1}..P{s+k}  sequence {args.reference[s:s+k]}")
        lines.append(f"  X-pattern: {minimal_pattern}")
        lines.append("")
    # Greedy (conservative) recommended pattern
    if chosen_cons:
        greedy_pattern = ''.join(args.reference[i] if i in set(chosen_cons) else 'X' for i in range(len(args.reference)))
        pretty_cons = ', '.join(f"P{i+1}:{args.reference[i]}" for i in chosen_cons)
        lines.append("Recommended specificity using greedy set (conservative):")
        lines.append(f"  Positions: {[p+1 for p in chosen_cons]}")
        lines.append(f"  Residues to define: {pretty_cons}")
        lines.append(f"  X-pattern: {greedy_pattern}")
        lines.append("")
    lines.append("Notes:")
    lines.append("- Conservative treats unknown residues ('?') as potentially matching, leading to stricter (sometimes impossible) uniqueness criteria.")
    lines.append("- Practical treats unknown residues as mismatches, providing actionable targets with available information.")

    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write('\n'.join(lines) + '\n')
    print(f"Specificity report written to {report_path}", file=sys.stderr)

    # Render figure with selection highlighting
    if args.highlight_mode == 'window' and win_prac is not None:
        s, k = win_prac
        chosen_positions = list(range(s, s + k))
    else:
        chosen_positions = chosen_prac
    selection_order = {p: i + 1 for i, p in enumerate(chosen_positions)}
    print(f"Rendering figure to {args.output} …", file=sys.stderr)
    plot_alignment(args.reference, group_isoforms(rows), args.output, context_map, chosen_positions, selection_order)

    # Optional: generate substitution specificity matrix figure
    if args.matrix_output:
        print(f"Rendering substitution specificity matrix to {args.matrix_output} …", file=sys.stderr)
        # Define alphabet (20 amino acids)
        aa_order = list("ACDEFGHIKLMNPQRSTVWY")
        ref = args.reference
        # Positions considered fixed for specificity (only these columns will be shown)
        fixed_positions = list(sorted(set(chosen_positions)))
        if not fixed_positions:
            print("No fixed positions selected; skipping matrix.", file=sys.stderr)
        else:
            import numpy as np
            Lm = len(fixed_positions)
            mat = np.zeros((len(aa_order), Lm), dtype=int)
            # For each fixed position and amino acid, test if choosing that residue remains unique vs all windows_considered
            for col_idx, j in enumerate(fixed_positions):
                for r_idx, aa in enumerate(aa_order):
                    # Build candidate residues at fixed positions: same as ref except at j set to aa
                    candidate_at_fixed = {p: (aa if p == j else ref[p]) for p in fixed_positions}
                    # Conflict if there exists a hit window w such that for all fixed positions p,
                    # w[p] equals candidate_at_fixed[p] (treat '?' or 'X' as potentially matching)
                    conflict = False
                    for w in windows_considered:
                        ok = True
                        for p, raa in candidate_at_fixed.items():
                            hp = w[p]
                            if hp == '?' or (isinstance(hp, str) and hp.upper() == 'X'):
                                continue
                            if isinstance(hp, str) and hp.upper() != raa.upper():
                                ok = False
                                break
                        if ok:
                            conflict = True
                            break
                    mat[r_idx, col_idx] = -1 if conflict else 1
            # Plot matrix only for fixed positions
            fig_h = max(4, 0.5 * len(aa_order) + 1.5)
            fig_w = max(6, 0.6 * Lm + 2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            cmap = plt.get_cmap('RdYlGn')
            im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
            ax.set_yticks(range(len(aa_order)))
            ax.set_yticklabels(aa_order)
            ax.set_xticks(range(Lm))
            ax.set_xticklabels([f"P{p+1}\n{ref[p]}" for p in fixed_positions])
            ax.set_title("Specificity by residue at selected positions (green = specific, red = cross-reactive)")
            # Mark the reference residue at each fixed position
            for col_idx, j in enumerate(fixed_positions):
                ref_aa = ref[j].upper()
                if ref_aa in aa_order:
                    row_idx = aa_order.index(ref_aa)
                    ax.text(col_idx, row_idx, '•', ha='center', va='center', color='black')
            fig.tight_layout()
            fig.savefig(args.matrix_output, dpi=200)
            plt.close(fig)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()



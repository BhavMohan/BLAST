#!/usr/bin/env python3
import io
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import streamlit as st


# =========================
# Data structures & parsing
# =========================

@dataclass
class BlastHit:
    header: str
    accession: Optional[str]
    start_pos: Optional[int]
    end_pos: Optional[int]
    organism: Optional[str]
    description: Optional[str]
    fragment_seq: str


def parse_header(header: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str], Optional[str]]:
    h = header.strip()
    if h.startswith(">"):
        h = h[1:]
    org_m = re.search(r"\[(.*?)\]\s*$", h)
    organism = org_m.group(1) if org_m else None
    core = h[: org_m.start()].strip() if org_m else h
    m = re.match(r"^(\S+?)(?::(\d+)-(\d+))?(?:\s+(.*))?$", core)
    accession = start = end = desc = None
    if m:
        accession = m.group(1)
        if m.group(2) and m.group(3):
            try:
                start = int(m.group(2)); end = int(m.group(3))
            except Exception:
                start = end = None
        desc = (m.group(4) or "").strip() or None
    return accession, start, end, organism, desc


def parse_fasta_clusters_text(text: str) -> List[BlastHit]:
    hits: List[BlastHit] = []
    cur_h = None; seq: List[str] = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cur_h is not None and seq:
                acc, s, e, org, desc = parse_header(cur_h)
                hits.append(BlastHit(cur_h.strip(), acc, s, e, org, desc, "".join(seq)))
            cur_h = line; seq = []
        else:
            if line:
                seq.append(line.strip())
    if cur_h is not None and seq:
        acc, s, e, org, desc = parse_header(cur_h)
        hits.append(BlastHit(cur_h.strip(), acc, s, e, org, desc, "".join(seq)))
    return hits


def filter_by_species(hits: List[BlastHit], species: str) -> List[BlastHit]:
    """More permissive species filter:
    - If species is empty or 'all'/'any' -> return all hits
    - Otherwise do case-insensitive 'contains' match on organism and full header
    - Special-case synonyms: 'Homo sapiens' ~= 'human', 'Mus musculus' ~= 'mouse'
    """
    s = (species or "").lower().strip()
    if not s or s in {"all", "any"}:
        return hits

    def match_text(text: str) -> bool:
        if not text:
            return False
        if s in text or text in s:
            return True
        if ("homo sapiens" in s and "human" in text) or ("human" in s and "homo sapiens" in text):
            return True
        if ("mus musculus" in s and "mouse" in text) or ("mouse" in s and "mus musculus" in text):
            return True
        return False

    def hit_matches(h: BlastHit) -> bool:
        o = (h.organism or "").lower().strip()
        header = (h.header or "").lower()
        return match_text(o) or match_text(header)

    return [h for h in hits if hit_matches(h)]


# =========================
# Alignment & specificity
# =========================

def score_alignment_at(ref: str, frag: str, start_idx: int) -> int:
    score = 0
    for i, aa in enumerate(frag):
        j = start_idx + i
        if j < 0 or j >= len(ref):
            return -10_000
        if aa.upper() == 'X':
            continue
        score += 1 if aa.upper() == ref[j].upper() else -1
    return score


def best_alignment_to_reference(ref: str, frag: str) -> Tuple[int, int]:
    best_s = 0; best = -10_000
    for s in range(0, len(ref) - len(frag) + 1):
        sc = score_alignment_at(ref, frag, s)
        if sc > best:
            best = sc; best_s = s
    return best_s, best


def build_alignment_rows(ref: str, hits: List[BlastHit]):
    rows = []
    for h in hits:
        s, score = best_alignment_to_reference(ref, h.fragment_seq)
        matches = 0; eff = 0
        for i, aa in enumerate(h.fragment_seq):
            j = s + i
            if 0 <= j < len(ref) and aa.upper() != 'X':
                eff += 1
                if aa.upper() == ref[j].upper():
                    matches += 1
        ident = (matches/eff) if eff else 0.0
        rows.append({"hit": h, "ref_start": s, "score": score, "identity": ident})
    rows.sort(key=lambda r: (-int(r["score"]), -float(r["identity"]), -len(r["hit"].fragment_seq)))
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


def build_mapped_window(ref: str, row: Dict[str, object]) -> str:
    # Map only the aligned fragment onto the reference; use '?' elsewhere
    h: BlastHit = row["hit"]  # type: ignore
    start_in_ref: int = int(row["ref_start"])  # 0-based
    L = len(ref)
    window = ["?"] * L
    frag = h.fragment_seq
    for i, aa in enumerate(frag):
        j = start_in_ref + i
        if 0 <= j < L:
            window[j] = aa
    return "".join(window)


def discriminative_positions(ref: str, windows: List[str], avoid_positions_1b: List[int], mode: str) -> Tuple[List[int], List[int]]:
    # Returns (chosen_positions 0-based, uncovered_hits_indices)
    L = len(ref)
    N = len(windows)
    remaining = set(range(N))
    chosen: List[int] = []
    # Precompute coverage per position
    cover: List[set] = []
    for i in range(L):
        cov = set()
        for h in range(N):
            aa = windows[h][i]
            if (aa == '?' or aa.upper() == 'X') and mode == 'conservative':
                continue
            if aa != '?' and aa.upper() == ref[i].upper():
                continue
            cov.add(h)
        cover.append(cov)

    # Build avoid set (0-based; -1 means last)
    avoid: set = set()
    for val in avoid_positions_1b:
        idx0 = (L - 1) if val == -1 else (val - 1)
        if 0 <= idx0 < L:
            avoid.add(idx0)

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
                    if '?' in hit_sub:
                        found_match = True
                        break
                    if all(a.upper() == b.upper() for a, b in zip(hit_sub, ref_sub)):
                        found_match = True
                        break
                else:
                    if '?' in hit_sub:
                        continue
                    if all(a.upper() == b.upper() for a, b in zip(hit_sub, ref_sub)):
                        found_match = True
                        break
            if not found_match:
                return (s, k)
    return None


# ================
# Plotting helper
# ================

def plot_alignment(ref: str, rows, out: io.BytesIO, chosen_positions=None, selection_order=None):
    if chosen_positions is None: chosen_positions = []
    if selection_order is None: selection_order = {}
    num_rows = len(rows)
    ref_len = len(ref)
    total_rows = num_rows + 1
    fig_h = max(3, 0.3 * total_rows + 1.8)
    fig, ax = plt.subplots(figsize=(max(8, ref_len), fig_h))

    def label_for_row(r: Dict[str, object]) -> str:
        h: BlastHit = r["hit"]  # type: ignore
        label = r.get("group_base_name") or (h.description or (h.accession or "(no accession)"))
        cnt = int(r.get("group_count", 1))
        if cnt > 1:
            label = f"{label}  (+{cnt-1} isoform(s))"
        return str(label)

    labels = [f"Reference — {ref}"] + [label_for_row(r) for r in rows]
    y_positions = [total_rows - i - 1 for i in range(total_rows)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(ref_len))
    ax.set_xticklabels([f"{i+1}\n{aa}" for i, aa in enumerate(ref)])

    # reference row
    y_ref = total_rows - 1
    for x, aa in enumerate(ref):
        rect = plt.Rectangle((x - 0.45, y_ref - 0.4), 0.9, 0.8, facecolor="#cfe8ff", edgecolor="#6ba6ff")
        ax.add_patch(rect); ax.text(x, y_ref, aa, ha='center', va='center', fontsize=9, color='#1f3b70')

    # hits
    for idx, row in enumerate(rows):
        h = row['hit']; start = int(row['ref_start']); frag = h.fragment_seq
        y = total_rows - (idx + 2)
        for x in range(ref_len):
            i = x - start
            aa = frag[i] if 0 <= i < len(frag) else None
            if aa is None:
                color = "#f9c0c0"
            else:
                if aa.upper() == 'X':
                    color = "#dddddd"
                else:
                    color = "#c6e48b" if aa.upper() == ref[x].upper() else "#f9c0c0"
            highlight = x in set(chosen_positions)
            edge = "#000000" if highlight else "#aaaaaa"
            lw = 1.8 if highlight else 1.0
            rect = plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, facecolor=color, edgecolor=edge, linewidth=lw)
            ax.add_patch(rect)
            if aa is not None:
                ax.text(x, y, aa, ha='center', va='center', fontsize=9, color='#222')

    ax.set_xlim(-1.0, ref_len + 10)
    ax.set_ylim(-0.5, total_rows - 0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ==========
# Streamlit
# ==========

st.set_page_config(page_title="Standalone BLAST specificity", layout="wide")
st.title("BLAST specificity — standalone single file")

with st.sidebar:
    st.markdown("Input parameters")
    ref = st.text_input("Peptide sequence", value="ALHGGWTTK")
    inward_text = st.text_input("Inward-facing positions (avoid; 1-based, comma; -1 = last)", value="2,-1")
    species = st.text_input("Species filter (partial allowed; 'all' for no filter)", value="Homo sapiens")
    mode = st.selectbox("Specificity mode", options=["conservative", "practical"], index=1)
    exclude_x = st.checkbox("Exclude hits containing 'X'", value=True)

uploaded = st.file_uploader("Upload NCBI 'FASTA (aligned clusters)' file", type=["txt","fa","fasta"]) 
run_btn = st.button("Run analysis")

if run_btn and uploaded and ref:
    text = uploaded.read().decode("utf-8", errors="ignore")
    all_hits = parse_fasta_clusters_text(text)
    # Precompute organism diagnostics
    org_counts: Dict[str, int] = {}
    for h in all_hits:
        k = h.organism or "(unknown)"
        org_counts[k] = org_counts.get(k, 0) + 1

    species_hits = filter_by_species(all_hits, species.strip()) if species.strip() else all_hits
    hits = [h for h in species_hits if 'X' not in h.fragment_seq.upper()] if exclude_x else species_hits

    if not hits:
        st.error(f"No hits after filtering (species='{species}', exclude_X={exclude_x}).")
        st.write(f"Counts — total: {len(all_hits)}, species-matched: {len(species_hits)}, after X-filter: {len(hits)}")
        if org_counts:
            top = sorted(org_counts.items(), key=lambda kv: -kv[1])[:10]
            diag_lines = [f"{name}: {count}" for name, count in top]
            st.info("Organisms detected in file (top 10):\n" + "\n".join(diag_lines))
        # Show sample headers to help choose filter text
        sample_headers = [h.header for h in all_hits[:5]]
        if sample_headers:
            st.text("\n".join(sample_headers))
        st.caption("Tip: set Species filter to 'all' or try a partial like 'sapiens' or 'human'. Uncheck 'Exclude hits containing X' if your file includes ambiguous residues.")
    else:
        rows = build_alignment_rows(ref, hits)
        rows = group_isoforms(rows)

        # Build mapped windows and filter out identicals
        windows_all = [build_mapped_window(ref, r) for r in rows]
        considered_rows = []
        windows_considered = []
        identical_count = 0
        for r, win in zip(rows, windows_all):
            h = r["hit"]
            if len(h.fragment_seq) == len(ref) and h.fragment_seq.upper() == ref.upper() and int(r["ref_start"]) == 0:
                identical_count += 1
                continue
            considered_rows.append(r)
            windows_considered.append(win)

        # Parse inward-facing positions (avoid first)
        avoid_1b: List[int] = []
        for tok in [t.strip() for t in inward_text.split(',') if t.strip()]:
            try:
                avoid_1b.append(int(tok))
            except ValueError:
                pass

        # Minimal residue set recommendation via greedy coverage on practical mode
        chosen_cons, uncovered_cons = discriminative_positions(ref, windows_considered, avoid_1b, 'conservative')
        chosen_prac, uncovered_prac = discriminative_positions(ref, windows_considered, avoid_1b, 'practical')
        win_cons = min_unique_window(ref, windows_considered, 'conservative')
        win_prac = min_unique_window(ref, windows_considered, 'practical')

        # Build text report
        lines: List[str] = []
        lines.append(f"Reference: {ref}")
        lines.append(f"{species} hits (after removing X): {len(considered_rows)}")
        if identical_count:
            lines.append(f"Identical to reference (ignored for specificity): {identical_count}")
        lines.append("")
        lines.append("Discriminative positions (P = 1-based positions):")
        lines.append(f"  Conservative: {[p+1 for p in chosen_cons]}  (uncovered hits: {len(uncovered_cons)})")
        if chosen_cons:
            aa_cons = ''.join(ref[i] for i in chosen_cons)
            pretty_cons = ', '.join(f"P{i+1}:{ref[i]}" for i in chosen_cons)
            lines.append(f"    Residues to define: {pretty_cons}  (motif: {aa_cons})")
        lines.append(f"  Practical:    {[p+1 for p in chosen_prac]}  (uncovered hits: {len(uncovered_prac)})")
        if chosen_prac:
            aa_prac = ''.join(ref[i] for i in chosen_prac)
            pretty_prac = ', '.join(f"P{i+1}:{ref[i]}" for i in chosen_prac)
            lines.append(f"    Residues to define: {pretty_prac}  (motif: {aa_prac})")
        lines.append("")
        lines.append("Minimal contiguous unique window vs hits:")
        lines.append(f"  Conservative: {('none' if win_cons is None else f'P{win_cons[0]+1}..P{win_cons[0]+win_cons[1]} sequence ' + ref[win_cons[0]:win_cons[0]+win_cons[1]])}")
        lines.append(f"  Practical:    {('none' if win_prac is None else f'P{win_prac[0]+1}..P{win_prac[0]+win_prac[1]} sequence ' + ref[win_prac[0]:win_prac[0]+win_prac[1]])}")
        lines.append("")
        if win_prac is not None:
            s, k = win_prac
            minimal_pattern = ''.join(ref[i] if (i >= s and i < s + k) else 'X' for i in range(len(ref)))
            lines.append("Recommended specificity using minimal window (practical):")
            lines.append(f"  Positions: P{s+1}..P{s+k}  sequence {ref[s:s+k]}")
            lines.append(f"  X-pattern: {minimal_pattern}")
            lines.append("")
        if chosen_prac:
            greedy_pattern_prac = ''.join(ref[i] if i in set(chosen_prac) else 'X' for i in range(len(ref)))
            pretty_prac = ', '.join(f"P{i+1}:{ref[i]}" for i in chosen_prac)
            lines.append("Recommended minimal greedy set (practical):")
            lines.append(f"  Positions: {[p+1 for p in chosen_prac]}")
            lines.append(f"  Residues to define: {pretty_prac}")
            lines.append(f"  X-pattern: {greedy_pattern_prac}")
            lines.append("")
        if chosen_cons:
            greedy_pattern = ''.join(ref[i] if i in set(chosen_cons) else 'X' for i in range(len(ref)))
            pretty_cons = ', '.join(f"P{i+1}:{ref[i]}" for i in chosen_cons)
            lines.append("Recommended specificity using greedy set (conservative):")
            lines.append(f"  Positions: {[p+1 for p in chosen_cons]}")
            lines.append(f"  Residues to define: {pretty_cons}")
            lines.append(f"  X-pattern: {greedy_pattern}")
            lines.append("")
        lines.append("Notes:")
        lines.append("- Conservative treats unknown residues ('?') as potentially matching, leading to stricter (sometimes impossible) uniqueness criteria.")
        lines.append("- Practical treats unknown residues as mismatches, providing actionable targets with available information.")

        report_txt = '\n'.join(lines) + '\n'

        # Figure
        if mode == 'practical' and win_prac is not None:
            s, k = win_prac
            chosen_positions = list(range(s, s + k))
        else:
            chosen_positions = chosen_prac
        selection_order = {p: i + 1 for i, p in enumerate(chosen_positions)}

        buf = io.BytesIO()
        plot_alignment(ref, considered_rows, buf, chosen_positions, selection_order)
        buf.seek(0)

        # Display & downloads
        st.subheader("Alignment figure")
        st.image(buf, caption="blast_alignment.png")
        st.download_button("Download blast_alignment.png", data=buf.getvalue(), file_name="blast_alignment.png", mime="image/png")

        st.subheader("Specificity report")
        st.code(report_txt)
        st.download_button("Download blast_specificity.txt", data=report_txt.encode("utf-8"), file_name="blast_specificity.txt", mime="text/plain")



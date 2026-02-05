import os, datetime
import re
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import hashlib
from pathlib import Path
import random
import time


from extract import  extract_psychological_evidence, ccd_from_psychological_evidence, extract_psychological_evidence_random, ccd_from_psychological_evidence_random
from compare_models import compare_cbt_models_embeddings_only


VERBOSE: bool = True

def _log(verbose: bool, msg: str):
    if verbose:
        print(msg)

_ST_CACHE = {}          
_TOKENIZER_CACHE = {} 

TEST = False

DATA_PATH = "publication_metadata_volumn1_filtered.csv"
GOLD_PATH = "Patient_Psi_CM_Dataset.json"

SEP = ","
TEXT_COL_COMBINED = "search_text"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

TRANSCRIPTS_DIR = "Psychotherapy_Transcripts"
PATIENT_DESCRIPTIONS_PATH = "patient_descriptions.txt"

SUMMARY_COL = "summary"

MAX_TOKENS = 480


OUTPUT_DIR = "outputs_log/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGE_RANGE_BINS = [
    (0, 20, "0-20 years"),
    (21, 30, "21-30 years"),
    (31, 40, "31-40 years"),
    (41, 50, "41-50 years"),
    (51, 60, "51-60 years"),
    (61, 70, "61-70 years"),
    (71, 80, "71-80 years"),
    (81, 120, "81 years plus"),
]


def _get_st_model(model_name: str, device: str = "cuda"):
    if model_name not in _ST_CACHE:
        _ST_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    return _ST_CACHE[model_name]

def _get_tokenizer(model_name: str):
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]

def _texts_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        if t is None:
            t = ""
        b = (t + "\n<SEP>\n").encode("utf-8", errors="ignore")
        h.update(b)
    return h.hexdigest()

def load_dataset(path: str, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)

def load_gold_situations(gold_path: str = GOLD_PATH) -> Dict[Tuple[str, str], str]:
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_map: Dict[Tuple[str, str], str] = {}
    for obj in data:
        name = str(obj.get("name", "") or "").strip()
        pid  = str(obj.get("id", "") or "").strip()
        sit  = str(obj.get("situation", "") or "").strip()
        if name and pid:
            gold_map[(name, pid)] = sit
    return gold_map

def load_gold_models(gold_path: str = GOLD_PATH) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for obj in data:
        name = str(obj.get("name", "") or "").strip()
        pid  = str(obj.get("id", "") or "").strip()
        if name and pid:
            gold_map[(name, pid)] = obj
    return gold_map

def get_gold_model(gold_map, patient_name: str, patient_id: str) -> Dict[str, Any]:
    return gold_map.get((patient_name, patient_id), {})

def build_search_text(row) -> str:
    title = str(row.get("Real_Title", "") or "")
    subjects = str(row.get("Psyc_Subjects", "") or "")
    symptoms = str(row.get("Symptoms", "") or "")

    parts = []
    if title.strip():
        parts.append(title.strip())
    if subjects.strip():
        parts.append(f"Subjects: {subjects.strip()}")
    if symptoms.strip():
        parts.append(f"Symptoms: {symptoms.strip()}")
    return ". ".join(parts)

def add_combined_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df[TEXT_COL_COMBINED] = df.apply(build_search_text, axis=1)
    return df

def add_summary_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if SUMMARY_COL not in df.columns:
        df[SUMMARY_COL] = ""

    df[SUMMARY_COL] = df[SUMMARY_COL].fillna("").astype(str)
    return df


def _chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return [""]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
    return chunks


def _encode_long_text(text: str, model, tokenizer, normalize: bool = True):
    chunks = _chunk_text(text, tokenizer)
    chunk_embs = model.encode(chunks, normalize_embeddings=False, show_progress_bar=False)
    doc_emb = np.mean(chunk_embs, axis=0)

    if normalize:
        norm = np.linalg.norm(doc_emb)
        if norm > 0:
            doc_emb = doc_emb / norm
    return doc_emb

def compute_case_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    cache_dir: str = "cache_embeddings",
    device: str = "cpu",
):
    """
    Ritorna (model, embeddings_normalizzati).
    Usa cache su disco per non ricalcolare embeddings dello stesso set di testi.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    model = _get_st_model(model_name, device=device)
    tokenizer = _get_tokenizer(model_name)

    fp = _texts_fingerprint(texts)
    cache_path = Path(cache_dir) / f"emb_{model_name.replace('/','__')}_{fp}.npy"

    if cache_path.exists():
        embeddings = np.load(cache_path)
        return model, embeddings

    all_embs = []
    for t in texts:
        emb = _encode_long_text(t, model=model, tokenizer=tokenizer, normalize=True)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)  # sicurezza shape
        all_embs.append(emb)

    embeddings = np.vstack(all_embs).astype(np.float32)
    np.save(cache_path, embeddings)
    return model, embeddings


def load_transcript_for_row(row, base_dir=TRANSCRIPTS_DIR) -> str:
    fname = str(row.get("file_name") or "").strip()
    if not fname:
        return ""
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def map_age_to_range(age: int) -> Optional[str]:
    for low, high, label in AGE_RANGE_BINS:
        if low <= age <= high:
            return label
    return None

def extract_demographics(description: str) -> dict:
    text = (description or "").lower()

    age = None
    age_patterns = [
        r"(\d{1,2})\s*-\s*year-old",
        r"(\d{1,2})\s*year-old",
        r"(\d{1,2})\s*years? old",
        r"aged\s+(\d{1,2})",
        r"age\s+(\d{1,2})",
    ]
    for pat in age_patterns:
        m = re.search(pat, text)
        if m:
            try:
                age_val = int(m.group(1))
                if 5 <= age_val <= 100:
                    age = age_val
                    break
            except ValueError:
                pass

    age_range = map_age_to_range(age) if age is not None else None

    gender = None
    if any(w in text for w in ["female", "woman", "girl"]):
        gender = "Female"
    elif any(w in text for w in ["male", "man", "boy"]):
        gender = "Male"
    elif "non-binary" in text or "nonbinary" in text:
        gender = "Non-binary"


    orientation = None
    if any(w in text for w in ["heterosexual", "straight"]):
        orientation = "Heterosexual"
    elif "lesbian" in text:
        orientation = "Lesbian"
    elif "bisexual" in text or "bi-sexual" in text:
        orientation = "Bisexual"
    elif "pansexual" in text:
        orientation = "Pansexual"
    elif "asexual" in text:
        orientation = "Asexual"
    elif "gay" in text or "homosexual" in text:
        orientation = "Gay"

    marital_status = None
    if "married" in text:
        marital_status = "Married"
    elif "single" in text:
        marital_status = "Single"
    elif "divorced" in text:
        marital_status = "Divorced"
    elif "widowed" in text or "widower" in text:
        marital_status = "Widowed"
    elif "separated" in text:
        marital_status = "Separated"
    elif "engaged" in text:
        marital_status = "Engaged"
    elif "in a relationship" in text or "has a partner" in text or "has a boyfriend" in text or "has a girlfriend" in text:
        marital_status = "In a relationship"

    return {
        "age": age,
        "age_range": age_range,
        "gender": gender,
        "orientation": orientation,
        "marital_status": marital_status,
    }

def split_by_demographics(
    results: pd.DataFrame,
    *,
    age_range: str | None = None,
    gender: str | None = None,
    orientation: str | None = None,
    marital_status: str | None = None,
):
    def row_is_full_match(row) -> bool:
        def field_match(user_val: str | None, col_name: str) -> bool:
            if user_val is None:
                return True
            ds_val = str(row.get(col_name, "") or "").strip().lower()
            user_norm = user_val.strip().lower()
            if not ds_val:
                return False
            return ds_val == user_norm

        return (
            field_match(age_range, "Client_Age_Range")
            and field_match(gender, "Client_Gender")
            and field_match(orientation, "Client_Sexual_Orientation")
            and field_match(marital_status, "Client_Marital_Status")
        )

    mask_full = results.apply(row_is_full_match, axis=1)
    return results[mask_full].copy(), results[~mask_full].copy()


class CaseRetriever:
    def __init__(self, df: pd.DataFrame, text_column: str = SUMMARY_COL):
        self.df = df
        self.text_column = text_column
        self.model, self.embeddings = compute_case_embeddings(
            df[text_column].fillna("").astype(str).tolist(),
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir="cache_embeddings",
            device="cuda",   
        )

    def search_embeddings(self, description: str, top_k: int = 10, threshold: float = 0.25):
        query_emb = self.model.encode([description], normalize_embeddings=True)[0]
        sims = self.embeddings @ query_emb
        idx_sorted = np.argsort(-sims)[:top_k]
        valid_idx = [i for i in idx_sorted if sims[i] >= threshold]

        results = self.df.iloc[valid_idx].copy()
        results["similarity"] = [float(sims[i]) for i in valid_idx]
        return results.sort_values("similarity", ascending=False)



def rerank_by_summary_similarity(df_with_summaries: pd.DataFrame, description: str, model: SentenceTransformer):
    query_emb = model.encode([description], normalize_embeddings=True)[0]
    summaries = df_with_summaries["Transcript_Summary"].fillna("").tolist()
    summary_embs = model.encode(summaries, normalize_embeddings=True, show_progress_bar=False)
    sims = summary_embs @ query_emb

    out = df_with_summaries.copy()
    out["summary_similarity"] = sims.astype(float)
    return out.sort_values(["summary_similarity", "similarity"], ascending=[False, False]).reset_index(drop=True)



def load_patient_descriptions(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: Dict[str, Dict[str, str]] = {}

    header_re = re.compile(r"^(?P<name>.+?)\s*\(ID:\s*(?P<pid>[^)]+)\)\s*$")

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        m = header_re.match(lines[0].strip())
        if not m:
            continue
        name = m.group("name").strip()
        pid = m.group("pid").strip()
        desc = "\n".join(lines[1:]).strip()
        out[name] = {"patient_id": pid, "description": desc}

    return out

def get_gold_situation(gold_map, patient_name: str, patient_id: str) -> str:
    gold = gold_map.get((patient_name, patient_id), "")
    return str(gold or "").strip()



def save_cbt_models_jsonl(
    out_path: str,
    *,
    patient_name: str,
    patient_id: str,
    items: list[dict],
):
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "patient_name": patient_name,
                "patient_id": patient_id,
                "source_text_id": it.get("source_text_id", ""),
                "retrieval_similarity": it.get("similarity", None),
                "summary_similarity": it.get("summary_similarity", None),
                "cbt_model": json.dumps(
                    it.get("cbt_model", {}),
                    ensure_ascii=False,
                    indent=2
                ),
                "cbt_eval": json.dumps(
                    it.get("cbt_eval", {}),
                    ensure_ascii=False,
                    indent=2
                ),
            }

            # UNA SOLA RIGA → JSONL valido
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_mistral_summary_for_selected_first(
    selected: pd.DataFrame,
    mistral_summaries_csv_path: str = "mistral_summaries.csv",
    file_col: str = "file_name",
    summary_col: str = "summary",
) -> str:
    """
    Prende il file_name del primo elemento di `selected`, legge mistral_summaries.csv,
    cerca la riga corrispondente e ritorna il valore della colonna `summary`.

    Assunzioni:
    - selected è un DataFrame non vuoto
    - mistral_summaries.csv contiene colonne: file_name, summary
    """

    if selected is None or selected.empty:
        return ""

    target_fname = str(selected.iloc[0].get(file_col, "") or "").strip()
    if not target_fname:
        return ""

    ms = pd.read_csv(mistral_summaries_csv_path, sep=None, engine="python")

    if file_col not in ms.columns:
        raise KeyError(f"Colonna '{file_col}' non trovata in {mistral_summaries_csv_path}. Colonne: {list(ms.columns)}")
    if summary_col not in ms.columns:
        raise KeyError(f"Colonna '{summary_col}' non trovata in {mistral_summaries_csv_path}. Colonne: {list(ms.columns)}")

    ms[file_col] = ms[file_col].astype(str).str.strip()
    match = ms.loc[ms[file_col] == target_fname, summary_col]

    if match.empty:
        return ""

    return str(match.iloc[0] or "").strip()




def set_summaries_from_csv(
    summaries_csv_path: str,
    data_path: str = DATA_PATH,
    sep: str = SEP,
) -> pd.DataFrame:
    df = load_dataset(data_path, sep=sep)

    if SUMMARY_COL not in df.columns:
        df[SUMMARY_COL] = ""

    summaries_df = pd.read_csv(
        summaries_csv_path,
        sep=None,              
        engine="python",       
    )

    df["file_name_norm"] = df["file_name"].astype(str).str.strip()
    summaries_df["file_name_norm"] = summaries_df["file_name"].astype(str).str.strip()

    summaries_df = summaries_df.rename(columns={"summary": "summary_from_csv"})

    merged = df.merge(
        summaries_df[["file_name_norm", "summary_from_csv"]],
        on="file_name_norm",
        how="left",
    )

    mask = merged["summary_from_csv"].notna()
    merged.loc[mask, SUMMARY_COL] = (
        merged.loc[mask, "summary_from_csv"].astype(str).str.strip()
    )

    updated_count = int(mask.sum())
    _log(VERBOSE, f"Aggiornate {updated_count} righe con summary da '{summaries_csv_path}'.")

    merged = merged.drop(columns=["file_name_norm", "summary_from_csv"])

    merged.to_csv(data_path, sep=sep, index=False)

    return merged



def run_pipeline_for_patient(description_p: str, description_id: str, patient_name: str = None) -> dict:
    """
    description -> transcript retrieval -> psychological report -> CCD
    """
    log_lines = []
    def _log_capture(enabled: bool, msg: str):
        if not enabled:
            return
        line = str(msg)
        print(line)          
        log_lines.append(line)  

    if patient_name is not None:
        _log_capture(VERBOSE, f"Using provided patient_name: {patient_name}")
        pdict = load_patient_descriptions(PATIENT_DESCRIPTIONS_PATH)
        _log_capture(VERBOSE, f"available_patients={len(pdict)} -> {sorted(list(pdict.keys()))}")

    

        if patient_name not in pdict:
            raise ValueError(f"Patient '{patient_name}' not found in {PATIENT_DESCRIPTIONS_PATH}. "
                            f"Available: {sorted(list(pdict.keys()))}")

        patient_id = pdict[patient_name]["patient_id"]
        description = pdict[patient_name]["description"]

    else:
        description = description_p.strip() #base case

    _log_capture(VERBOSE, f"Using description: {description} with ID={description_id}")
    demo = extract_demographics(description)

    df = load_dataset(DATA_PATH, sep=SEP)
    df = add_combined_text_column(df)  
    df = add_summary_text_column(df)
    retriever_summary = CaseRetriever(df, text_column=SUMMARY_COL)

    retriever_text_combined = CaseRetriever(df, text_column=TEXT_COL_COMBINED)


    results = retriever_summary.search_embeddings(description, top_k=10, threshold=0.50)

    if len(results) > 0 and results.iloc[0].get("similarity", 0) < 0.60:
        _log_capture(VERBOSE, "Low similarity from summary retriever, trying combined text retriever...")
        results = retriever_text_combined.search_embeddings(description, top_k=10, threshold=0.6)

    if len(results) > 0:
        _log_capture(VERBOSE, "Top results (by metadata similarity):")
        for i, r in results.head(10).iterrows():
            _log_capture(VERBOSE, f"  - sim={r['similarity']:.3f} | title='{str(r.get('Real_Title',''))[:80]}' | file='{r.get('file_name','')}'")
    else:
        results = retriever_summary.search_embeddings(description, top_k=10, threshold=0.10)
        _log_capture(VERBOSE, "Emergency Fallback to 0.10 - Top results (by metadata similarity):")
        for i, r in results.head(10).iterrows():
            _log_capture(VERBOSE, f"  - sim={r['similarity']:.3f} | title='{str(r.get('Real_Title',''))[:80]}' | file='{r.get('file_name','')}'")
  



    full, partial = split_by_demographics(
        results,
        age_range=demo["age_range"],
        gender=demo["gender"],
        orientation=demo["orientation"],
        marital_status=demo["marital_status"],
    )

    _log_capture(VERBOSE, f"\n=== DEMOGRAPHICS FILTER ===")
    _log_capture(VERBOSE, f"full_match={len(full)} partial_match={len(partial)}")
    _log_capture(VERBOSE, f"match_used={'FULL' if (full is not None and not full.empty) else 'PARTIAL'}")

    if full is not None and not full.empty:
        selected = full.sort_values("similarity", ascending=False).head(3).copy()
        match_kind = "FULL"
    else:
        selected = partial.sort_values("similarity", ascending=False).head(3).copy()
        match_kind = "PARTIAL"

    _log_capture(VERBOSE, f"\n=== SELECTED CANDIDATES ===")
    _log_capture(VERBOSE, f"selected_n={len(selected)}")
    for i, r in selected.iterrows():
        _log_capture(VERBOSE, f"  - sim={r['similarity']:.3f} | file='{r.get('file_name','')}' | title='{str(r.get('Real_Title',''))[:80]}'")



    row = selected.iloc[0]
    _log_capture(VERBOSE, f"Using top selected file for transcript: '{row.get('file_name','')}'")
    full_transcript = load_transcript_for_row(row, TRANSCRIPTS_DIR)

    report = extract_psychological_evidence(description, full_transcript)
    _log_capture(VERBOSE, f"\n PSYCHOLOGICAL REPORT FROM TRANSCRIPT")
    _log_capture(VERBOSE, report )

    summary_text = get_mistral_summary_for_selected_first(selected, "mistral_summaries.csv")
    _log_capture(VERBOSE, f"Summary da mistral_.csv: {summary_text}")

    refined_CCD = ccd_from_psychological_evidence(report, description)

    def _safe_list(x):
        return x if isinstance(x, list) else ([] if x is None else [x])

    _log_capture(VERBOSE, "\n REFINED CCD: description + report")
    output = {
    "history": (refined_CCD.get("history") or "").strip() if isinstance(refined_CCD.get("history"), str) else (refined_CCD.get("history") or ""),
    "helpless_belief": refined_CCD.get("helpless_belief", ""),
    "unlovable_belief": refined_CCD.get("unlovable_belief", ""),
    "worthless_belief": refined_CCD.get("worthless_belief", ""),
    "intermediate_belief": refined_CCD.get("intermediate_belief", ""),
    "intermediate_belief_depression": refined_CCD.get("intermediate_belief_depression", ""),
    "coping_strategies": _safe_list(refined_CCD.get("coping_strategies", [])),
    "situation": refined_CCD.get("situation", ""),
    "auto_thought": refined_CCD.get("auto_thought", ""),
    "emotion": refined_CCD.get("emotion", ""),
    "behavior": refined_CCD.get("behavior", ""),
}

    _log_capture(VERBOSE, json.dumps(output, indent=2, ensure_ascii=False))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, f"run_log_{patient_name or 'adhoc'}_{ts}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return refined_CCD
    

def run_pipeline_random(description_p: str, description_id: str, patient_name: str = None) -> dict:
    """
    Pipeline to run for selecting a random transcript from the folder.
    """

    log_lines = []

    def _log_capture(enabled: bool, msg: str):
        if not enabled:
            return
        line = str(msg)
        print(line)
        log_lines.append(line)

    if patient_name is not None:
        _log_capture(VERBOSE, f"Using provided patient_name: {patient_name}")
        pdict = load_patient_descriptions(PATIENT_DESCRIPTIONS_PATH)
        _log_capture(VERBOSE, f"available_patients={len(pdict)} -> {sorted(list(pdict.keys()))}")

        if patient_name not in pdict:
            raise ValueError(
                f"Patient '{patient_name}' not found in {PATIENT_DESCRIPTIONS_PATH}. "
                f"Available: {sorted(list(pdict.keys()))}"
            )

        description = pdict[patient_name]["description"]
    else:
        description = description_p.strip()

    _log_capture(VERBOSE, f"Using description: {description} with ID={description_id}")

    if not os.path.isdir(TRANSCRIPTS_DIR):
        raise FileNotFoundError(f"TRANSCRIPTS_DIR not found or not a directory: {TRANSCRIPTS_DIR}")

    transcript_files = [
        fn for fn in os.listdir(TRANSCRIPTS_DIR)
        if os.path.isfile(os.path.join(TRANSCRIPTS_DIR, fn))
        and fn.lower().endswith((".txt", ".md", ".transcript"))
    ]

    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found in {TRANSCRIPTS_DIR}")

    chosen_file = random.choice(transcript_files)
    chosen_path = os.path.join(TRANSCRIPTS_DIR, chosen_file)
    _log_capture(VERBOSE, f"Randomly selected transcript file: '{chosen_file}'")

    with open(chosen_path, "r", encoding="utf-8") as f:
        full_transcript = f.read()

    if not full_transcript.strip():
        raise ValueError(f"Selected transcript is empty: {chosen_file}")

    report = extract_psychological_evidence_random(description, full_transcript)
    _log_capture(VERBOSE, f"\nPSYCHOLOGICAL REPORT FROM TRANSCRIPT")
    _log_capture(VERBOSE, report)


    refined_CCD = ccd_from_psychological_evidence(report, description)

    time.sleep(1.0)

    def _safe_list(x):
        return x if isinstance(x, list) else ([] if x is None else [x])

    output = {
        "history": (refined_CCD.get("history") or "").strip()
        if isinstance(refined_CCD.get("history"), str)
        else (refined_CCD.get("history") or ""),
        "helpless_belief": refined_CCD.get("helpless_belief", ""),
        "unlovable_belief": refined_CCD.get("unlovable_belief", ""),
        "worthless_belief": refined_CCD.get("worthless_belief", ""),
        "intermediate_belief": refined_CCD.get("intermediate_belief", ""),
        "intermediate_belief_depression": refined_CCD.get("intermediate_belief_depression", ""),
        "coping_strategies": _safe_list(refined_CCD.get("coping_strategies", [])),
        "situation": refined_CCD.get("situation", ""),
        "auto_thought": refined_CCD.get("auto_thought", ""),
        "emotion": refined_CCD.get("emotion", ""),
        "behavior": refined_CCD.get("behavior", ""),
        "source_text_id": chosen_file,
    }

    _log_capture(VERBOSE, json.dumps(output, indent=2, ensure_ascii=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, f"run_log_{patient_name or 'adhoc'}_{ts}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return output


def run_pipeline_for_patient_least_correletad(description_p: str, description_id: str, patient_name: str = None) -> dict:
    """
    Pipeline to run for selecting the least correlated case based on demographics and description.
    """

    log_lines = []

    def _log_capture(enabled: bool, msg: str):
        if not enabled:
            return
        line = str(msg)
        print(line)
        log_lines.append(line)

    if patient_name is not None:
        _log_capture(VERBOSE, f"Using provided patient_name: {patient_name}")
        pdict = load_patient_descriptions(PATIENT_DESCRIPTIONS_PATH)
        _log_capture(VERBOSE, f"available_patients={len(pdict)} -> {sorted(list(pdict.keys()))}")

        if patient_name not in pdict:
            raise ValueError(
                f"Patient '{patient_name}' not found in {PATIENT_DESCRIPTIONS_PATH}. "
                f"Available: {sorted(list(pdict.keys()))}"
            )

        description = pdict[patient_name]["description"]
    else:
        description = (description_p or "").strip()

    _log_capture(VERBOSE, f"Using description: {description} with ID={description_id}")

    demo = extract_demographics(description)

    df = load_dataset(DATA_PATH, sep=SEP)
    df = add_combined_text_column(df)
    df = add_summary_text_column(df)

    retriever_summary = CaseRetriever(df, text_column=SUMMARY_COL)
    retriever_text_combined = CaseRetriever(df, text_column=TEXT_COL_COMBINED)

    try:
        results = retriever_summary.search_embeddings(description, top_k=len(df), threshold=None)
    except TypeError:
        results = retriever_summary.search_embeddings(description, top_k=len(df), threshold=0.0)

    if results is None or len(results) == 0:
        _log_capture(VERBOSE, "No results from summary retriever (no-threshold). Trying combined text retriever...")
        try:
            results = retriever_text_combined.search_embeddings(description, top_k=len(df), threshold=None)
        except TypeError:
            results = retriever_text_combined.search_embeddings(description, top_k=len(df), threshold=0.0)

    if results is None or len(results) == 0:
        raise RuntimeError("Retriever returned no results even with no threshold. Cannot proceed.")

    _log_capture(VERBOSE, "Sample results (metadata similarity):")
    for _, r in results.sort_values("similarity", ascending=True).head(10).iterrows():
        _log_capture(
            VERBOSE,
            f"  - sim={r['similarity']:.3f} | title='{str(r.get('Real_Title',''))[:80]}' | file='{r.get('file_name','')}'"
        )

    full, partial = split_by_demographics(
        results,
        age_range=demo["age_range"],
        gender=demo["gender"],
        orientation=demo["orientation"],
        marital_status=demo["marital_status"],
    )

    _log_capture(VERBOSE, f"\n=== DEMOGRAPHICS FILTER ===")
    _log_capture(VERBOSE, f"full_match={len(full)} partial_match={len(partial)}")
    _log_capture(VERBOSE, f"match_used={'FULL' if (full is not None and not full.empty) else 'PARTIAL'}")

    if full is not None and not full.empty:
        pool = full
        match_kind = "FULL"
    else:
        pool = partial
        match_kind = "PARTIAL"

    if pool is None or pool.empty:
        _log_capture(VERBOSE, "WARNING: demographics filter removed all candidates. Falling back to unfiltered results.")
        pool = results
        match_kind = "NO_DEMO_FALLBACK"

    selected = pool.sort_values("similarity", ascending=True).head(3).copy() # to select low similarity

    _log_capture(VERBOSE, f"\n=== SELECTED CANDIDATES (LEAST RELATED) ===")
    _log_capture(VERBOSE, f"selected_n={len(selected)} match_kind={match_kind}")
    for _, r in selected.iterrows():
        _log_capture(
            VERBOSE,
            f"  - sim={r['similarity']:.3f} | file='{r.get('file_name','')}' | title='{str(r.get('Real_Title',''))[:80]}'"
        )
    row = selected.iloc[0]
    _log_capture(VERBOSE, f"Using LEAST related file for transcript: '{row.get('file_name','')}'")
    full_transcript = load_transcript_for_row(row, TRANSCRIPTS_DIR)

    report = extract_psychological_evidence_random(description, full_transcript)
    _log_capture(VERBOSE, f"\nPSYCHOLOGICAL REPORT FROM TRANSCRIPT")
    _log_capture(VERBOSE, report)

    summary_text = get_mistral_summary_for_selected_first(selected, "mistral_summaries.csv")
    _log_capture(VERBOSE, f"Summary da mistral_.csv: {summary_text}")

    refined_CCD = ccd_from_psychological_evidence_random(report, description)

    def _safe_list(x):
        return x if isinstance(x, list) else ([] if x is None else [x])

    output = {
        "history": (refined_CCD.get("history") or "").strip()
        if isinstance(refined_CCD.get("history"), str)
        else (refined_CCD.get("history") or ""),
        "helpless_belief": refined_CCD.get("helpless_belief", ""),
        "unlovable_belief": refined_CCD.get("unlovable_belief", ""),
        "worthless_belief": refined_CCD.get("worthless_belief", ""),
        "intermediate_belief": refined_CCD.get("intermediate_belief", ""),
        "intermediate_belief_depression": refined_CCD.get("intermediate_belief_depression", ""),
        "coping_strategies": _safe_list(refined_CCD.get("coping_strategies", [])),
        "situation": refined_CCD.get("situation", ""),
        "auto_thought": refined_CCD.get("auto_thought", ""),
        "emotion": refined_CCD.get("emotion", ""),
        "behavior": refined_CCD.get("behavior", ""),
    }

    _log_capture(VERBOSE, "\nREFINED CCD: description + report")
    _log_capture(VERBOSE, json.dumps(output, indent=2, ensure_ascii=False))


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, f"run_log_{patient_name or 'adhoc'}_{ts}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return output


if __name__ == "__main__":

    patient_name = ""
    description = """" History (facts):
- substance abuse treatment in the past
- long-term obesity
- childhood bullying
- rehab participation

Situation (facts):
- overslept on workday
- insurance payments stopped
- considering not going to work"""
    
    set_summaries_from_csv("mistral_summaries.csv")

    _log(VERBOSE, f"\nPIPELINE START ")
    if patient_name:
        _log(VERBOSE, f"patient_name='{patient_name}'")
    else:
        _log(VERBOSE, f"descriptions_file='{PATIENT_DESCRIPTIONS_PATH}'")

    ranked_df = run_pipeline_for_patient(description_p=description)

    print(ranked_df[["Real_Title", "source_text_id", "similarity", "summary_similarity", "match_kind"]].head(5))

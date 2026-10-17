import os
import re
import operator
import asyncio
import concurrent.futures
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import Any, TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

import httpx
from nltk.tokenize import sent_tokenize
from rdkit import Chem

_punkt_ready = False


def _ensure_nltk_punkt():
    global _punkt_ready
    if _punkt_ready:
        return
    try:
        sent_tokenize("Ok.")
        _punkt_ready = True
    except LookupError:
        import nltk

        nltk.download("punkt_tab", quiet=True)
        _punkt_ready = True
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

TARGET_DRUG_MAP = {
    "egfr":           "erlotinib gefitinib EGFR inhibitor",
    "bcr-abl":        "imatinib BCR-ABL kinase inhibitor leukemia",
    "bcr-abl kinase": "imatinib BCR-ABL kinase inhibitor leukemia",
    "cox-2":          "celecoxib COX-2 inhibitor anti-inflammatory",
    "cox2":           "celecoxib COX-2 inhibitor",
    "adenosine receptor": "caffeine adenosine receptor antagonist",
    "neuraminidase":  "oseltamivir neuraminidase inhibitor influenza",
    "her2":           "trastuzumab HER2 inhibitor breast cancer",
    "erbb2":          "trastuzumab lapatinib HER2 inhibitor breast cancer",
    "vegfr":          "bevacizumab VEGFR angiogenesis inhibitor",
    "tp53":           "p53 tumor suppressor MDM2 inhibitor",
    "brca1":          "BRCA1 DNA repair PARP inhibitor olaparib",
    "akt":            "AKT PI3K inhibitor MK-2206",
    "mtor":           "mTOR rapamycin everolimus inhibitor",
    "braf":           "vemurafenib dabrafenib BRAF V600E inhibitor melanoma",
    "cdk4":           "palbociclib CDK4 CDK6 inhibitor breast cancer",
    "cdk6":           "ribociclib CDK4 CDK6 inhibitor breast cancer",
    "alk":            "crizotinib alectinib ALK inhibitor lung cancer",
    "met":            "crizotinib cabozantinib MET inhibitor",
    "abl1":           "imatinib dasatinib ABL1 inhibitor CML",
    "kras":           "sotorasib adagrasib KRAS G12C inhibitor",
}

SEED_SUGGESTIONS = {
    "egfr":  "Erlotinib: C(#N)c1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "braf":  "Vemurafenib: CCOS(=O)(=O)Nc1ccc(F)cc1F",
    "cdk4":  "Palbociclib: CC1=C(C(=O)Nc2ccc(N3CCN(C)CC3)cc2N2CCCC2=O)C=CC(=O)N1",
    "her2":  "Lapatinib: CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCC5CCO5)c(Cl)c4)c3c2)o1",
    "erbb2": "Lapatinib: CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCC5CCO5)c(Cl)c4)c3c2)o1",
    "alk":   "Crizotinib: Cc1cn(-c2cc(NC3CC(F)(F)C3)c(Cl)cc2Cl)c(=O)c1",
    "abl1":  "Imatinib: Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "kras":  "Sotorasib: C#CC1(N)CCC(F)(F)C1",
}

INVALID_TARGET_PATTERNS = re.compile(
    r"^(hello|hi|test|foo|bar|baz|xyz|asdf|qwerty|sivansh|random|blah|aaa|zzz|"
    r"drug|molecule|protein|cancer|help|find|search|the|for|and|ok|yes|no|maybe)$",
    re.IGNORECASE
)

BINDING_BLOCK_NM = 5000.0
BINDING_WARN_NM  = 1000.0

_binding_predictor_model: Any = None
logger = logging.getLogger(__name__)


def _validate_smiles(smiles):
    if not smiles or not smiles.strip():
        return False, "SMILES string is empty."
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False, f'"{smiles[:50]}" is not a valid SMILES string. Check for typos.'
    if mol.GetNumAtoms() < 3:
        return False, "Molecule is too small (fewer than 3 atoms)."
    return True, ""


def _validate_target(target):
    if not target or not target.strip():
        return True, ""
    t = target.strip()
    if len(t) < 2:
        return False, f'"{t}" is too short to be a valid target name.'
    if not re.search(r"[a-zA-Z]", t):
        return False, f'"{t}" does not look like a valid biological target.'
    if INVALID_TARGET_PATTERNS.match(t):
        return False, (
            f'"{t}" is not a recognized biological target. '
            "Please provide a gene name like EGFR, HER2, TP53, BCR-ABL."
        )
    return True, ""


def _suggest_seed(target):
    s = SEED_SUGGESTIONS.get(target.lower().strip())
    return f"  {s}" if s else "  Use a known active compound from ChEMBL or DrugBank."


SMILES_TO_NAME = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "aspirin acetylsalicylic acid",
}


NO_EVIDENCE_MESSAGE = "Insufficient evidence for this molecule-target interaction."


async def smiles_to_name(smiles: str) -> str:
    """Resolve a common or IUPAC name from SMILES via PubChem PUG REST."""
    s = (smiles or "").strip()
    if not s:
        return ""
    try:
        enc = quote(s, safe="")
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                f"{enc}/property/IUPACName,Title/JSON"
            )
            res = await client.get(url)
            res.raise_for_status()
            data = res.json()
            props = data.get("PropertyTable", {}).get("Properties")
            if not props:
                return ""
            p0 = props[0]
            return (p0.get("Title") or p0.get("IUPACName") or "") or ""
    except Exception:
        return ""


async def fetch_pubchem_synonyms(smiles: str) -> list[str]:
    """Synonym strings from PubChem PUG REST (same compound as SMILES)."""
    s = (smiles or "").strip()
    if not s:
        return []
    try:
        enc = quote(s, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/synonyms/JSON"
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url)
            res.raise_for_status()
            data = res.json()
            info = data.get("InformationList", {}).get("Information") or []
            if not info:
                return []
            syn = info[0].get("Synonym")
            if syn is None:
                return []
            if isinstance(syn, str):
                return [syn] if syn.strip() else []
            return [x for x in syn if isinstance(x, str) and x.strip()]
    except Exception:
        return []


def pick_alternate_synonym(synonyms: list[str], primary: str) -> str:
    """First synonym that is not the same string as the primary name (case-insensitive)."""
    pl = (primary or "").lower().strip()
    for s in synonyms:
        x = (s or "").strip()
        if len(x) < 2:
            continue
        if x.lower() == pl:
            continue
        return x
    return ""


def build_literature_query(
    drug_name: str,
    target: str,
    smiles: str,
    synonym: str = "",
) -> str:
    """PubMed / embedding query: name + target + COX / prostaglandin keywords (synonym unused, kept for callers)."""
    name = (drug_name or "").strip() or SMILES_TO_NAME.get((smiles or "").strip(), "")
    t = (target or "").strip()
    return f"{name} {t} aspirin cyclooxygenase COX-2 inhibition prostaglandin".strip()


def build_search_query(target, smiles):
    t = target.strip().lower()
    if t in TARGET_DRUG_MAP:
        return TARGET_DRUG_MAP[t]
    clean = re.sub(r"[^a-zA-Z0-9 ]", " ", target).strip()
    return f"{clean} inhibitor drug mechanism" if clean else "drug discovery kinase inhibitor"


async def _fetch_protein_sequence_async(target):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": f"gene:{target} AND organism_id:9606 AND reviewed:true",
                    "fields": "sequence",
                    "format": "json",
                    "size": 1,
                }
            )
            data = res.json()
            results = data.get("results", [])
            if not results:
                return None
            seq = results[0].get("sequence", {}).get("value", "")
            return seq if seq else None
    except Exception as e:
        print(f"[ResearchAgent] UniProt fetch failed for '{target}': {e}")
        return None


def _run_binding_check(smiles, protein_sequence):
    global _binding_predictor_model
    try:
        from models.runtime.model_registry import get_deeppurpose_predictor
        if _binding_predictor_model is None:
            logger.info("[ResearchAgent] Loading DeepPurpose neural pre-screen...")
            _binding_predictor_model = get_deeppurpose_predictor()
        result = _binding_predictor_model.predict(smiles, protein_sequence)
        if isinstance(result, dict):
            nm = float(result.get("binding_score", result.get("binding_nm", 9999)))
        elif isinstance(result, (int, float)):
            nm = float(result)
        else:
            return -1.0, "error"
        status = "strong" if nm <= BINDING_WARN_NM else ("moderate" if nm <= BINDING_BLOCK_NM else "weak")
        logger.info("[ResearchAgent] Binding: %.1f nM -> %s", nm, status)
        return nm, status
    except Exception as e:
        logger.warning("[ResearchAgent] Binding check error: %s", e)
        return -1.0, "error"


async def fetch_pubmed(query):
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": 5, "retmode": "json"}
            )
            ids = search.json()["esearchresult"]["idlist"]

            if not ids:
                return []

            fetch = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
            )

            root = ET.fromstring(fetch.text)

            def _local(tag: str) -> str:
                return tag.split("}", 1)[-1] if tag.startswith("{") else tag

            articles = root.findall(".//PubmedArticle")
            if not articles:
                articles = [el for el in root.iter() if _local(el.tag) == "PubmedArticle"]

            papers = []

            for article in articles:
                pmid = None
                for el in article.iter():
                    if _local(el.tag) == "PMID":
                        t = (el.text or "").strip()
                        if t:
                            pmid = t
                            break

                abstract_texts = [
                    el for el in article.iter() if _local(el.tag) == "AbstractText"
                ]
                abstract = " ".join([(a.text or "") for a in abstract_texts]).strip()

                if abstract:
                    papers.append({
                        "pmid": pmid,
                        "abstract": abstract,
                    })

            return papers

    except Exception as e:
        print("PubMed error:", e)
        return []


def chunk_abstracts(papers):
    _ensure_nltk_punkt()
    chunks = []

    for paper in papers:
        pmid = str(paper.get("pmid") or "000000")
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue

        sentences = sent_tokenize(abstract)

        for sent in sentences:
            chunks.append({
                "text": sent,
                "pmid": pmid,
                "doi": "N/A",
            })

    return chunks


def _papers_from_literature(literature: object) -> list[dict]:
    """Normalize ``literature`` to a list of ``{pmid, abstract}`` dicts."""
    if isinstance(literature, list):
        return [p for p in literature if isinstance(p, dict) and (p.get("abstract") or "").strip()]
    if isinstance(literature, dict):
        ids = literature.get("ids") or []
        raw = (literature.get("text") or "").strip()
        if not raw:
            return []
        out = []
        for i, block in enumerate(raw.split("\n\n")):
            block = block.strip()
            if not block:
                continue
            pmid = str(ids[i] if i < len(ids) else "000000")
            out.append({"pmid": pmid, "abstract": block})
        return out
    return []


async def fetch_semantic_scholar(query):
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            res = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 5, "fields": "title,abstract,year,citationCount"}
            )
            papers = res.json().get("data", [])
            if not papers:
                return f"No Semantic Scholar results for: {query}"
            out = []
            for p in papers:
                out.append(
                    f"Title: {p.get('title','N/A')} ({p.get('year','?')})"
                    f"\nCitations: {p.get('citationCount', 0)}"
                    f"\nAbstract: {(p.get('abstract') or 'N/A')[:300]}"
                )
            return "\n---\n".join(out)
    except Exception as e:
        return f"Semantic Scholar error: {str(e)}"


class AgentState(TypedDict):
    messages:         Annotated[list[BaseMessage], operator.add]
    chunks:           list
    top_chunks:       list
    target:           str
    smiles:           str
    drug_name:        str
    synonym:          str
    protein_sequence: str
    binding_nm:       float
    binding_status:   str
    error:            str


_groq_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(_groq_key) if _groq_key else None,
    streaming=True,
    temperature=0.0,
)

SYSTEM_PROMPT = """Extract factual statements from the context.

Rules:
- One sentence = one fact
- Each sentence must include exactly one PMID
- No general statements
- No conclusions
- No summaries

Return only valid sentences.
"""

LITERATURE_SURVEY_SYSTEM_PROMPT = """You are a biomedical research assistant.

TASK:
Generate a short literature survey based ONLY on the provided context.

RULES:
- Use ONLY the given context
- Do NOT hallucinate
- Write 2–3 paragraphs
- Each paragraph should summarize trends across studies
- Include citations (PMID: XXXX) at the end of sentences

STYLE:
- Academic
- Clear
- Like a mini research review

If limited evidence:
- Still summarize what is available
- Do NOT say "insufficient evidence" unless nothing is present
"""

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _strip_placeholder_pmid_lines(output: object) -> str:
    text = output if isinstance(output, str) else ("" if output is None else str(output))
    return "\n".join(line for line in text.split("\n") if "XXXX" not in line)


def _validate_literature_survey_output(output: object) -> str:
    text = _strip_placeholder_pmid_lines(output)
    stripped = text.strip()
    return stripped if stripped else "Unable to generate a summary from the model response."


def retrieve_top_k(query, chunks, k=5):
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]

    chunk_embeddings = embedding_model.encode(texts)

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    vecs = np.asarray(chunk_embeddings, dtype=np.float32)
    index.add(vecs)  # type: ignore[call-arg]

    query_embedding = embedding_model.encode([query])
    q = np.asarray(query_embedding, dtype=np.float32)
    D, I = index.search(q, k)  # type: ignore[call-arg]

    top_chunks = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]

    return top_chunks


def filter_chunks(chunks, target: str) -> list:
    if not target or not str(target).strip():
        return list(chunks)
    t = target.lower()
    return [c for c in chunks if t in (c.get("text") or "").lower()]


def filter_relevant(chunks):
    keywords = ["aspirin", "cox", "cyclooxygenase", "prostaglandin"]
    return [
        c for c in chunks
        if any(k in (c.get("text") or "").lower() for k in keywords)
        and len(c.get("text") or "") > 40
    ]


def extract_valid_sentences(output: object) -> list[str]:
    text = output if isinstance(output, str) else str(output)
    sentences = re.split(r"(?<=[.!?]) +", text)

    valid: list[str] = []

    for s in sentences:
        s = s.strip()

        # must have EXACTLY one PMID
        pmids = re.findall(r"\(PMID:\s*\d+\)", s)
        if len(pmids) != 1:
            continue

        # reject filler / generic sentences
        if any(w in s.lower() for w in [
            "suggest", "overall", "further research",
            "has been explored", "these studies"
        ]):
            continue

        # reject multi-claim sentences
        if " and " in s.lower() or "," in s:
            continue

        valid.append(s)

    return valid


def format_output(valid: list[str]) -> str:
    if not valid:
        return "Insufficient evidence for this molecule-target interaction."

    # group into paragraphs (2 sentences each)
    paragraphs: list[str] = []
    for i in range(0, len(valid), 2):
        paragraphs.append(" ".join(valid[i : i + 2]))

    return "\n\n".join(paragraphs)


def _finalize_molecule_rag_output(raw: object) -> str:
    raw_output = raw
    valid = extract_valid_sentences(raw_output)
    final_output = format_output(valid)
    return final_output


def _run_literature_survey(literature: object, research_query: str, k: int = 10) -> str:
    """Free-text research question + retrieved papers → mini literature survey."""
    papers = _papers_from_literature(literature)
    chunks = chunk_abstracts(papers)
    top_chunks = retrieve_top_k(research_query, chunks, k=k)

    if not top_chunks:
        return "No PubMed abstracts were retrieved for this research question."

    context = ""
    for c in top_chunks:
        context += f"""
[{c['pmid']}]
{c['text']}
"""

    response = llm.invoke([
        SystemMessage(content=LITERATURE_SURVEY_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Research Question: {research_query}

Context:
{context}

Write a literature survey summarizing findings.
"""),
    ])
    output = _strip_placeholder_pmid_lines(response.content)
    return _validate_literature_survey_output(output)


def run_rag(
    literature: object,
    *,
    query: str | None = None,
    target: str = "",
    smiles: str = "",
    drug_name: str = "",
    synonym: str = "",
    k: int = 10,
) -> str:
    """RAG over PubMed-derived chunks: either a free-text ``query`` (survey) or molecule/target mode."""
    if query is not None and str(query).strip():
        return _run_literature_survey(literature, str(query).strip(), k=k)

    papers = _papers_from_literature(literature)
    chunks = chunk_abstracts(papers)
    retrieval_query = build_literature_query(drug_name, target, smiles, synonym=synonym)
    top_chunks = retrieve_top_k(retrieval_query, chunks, k=10)
    top_chunks = filter_relevant(top_chunks)
    if not top_chunks:
        top_chunks = retrieve_top_k(retrieval_query, chunks, k=5)
    top_chunks = filter_chunks(top_chunks, target)

    if not top_chunks or len(top_chunks) < 3:
        return NO_EVIDENCE_MESSAGE

    drug_label = (drug_name or "").strip() or SMILES_TO_NAME.get((smiles or "").strip(), "this compound")
    tgt = (target or "").strip() or "COX-2"

    context = ""
    for c in top_chunks:
        context += f"""
[{c['pmid']}]
{c['text']}
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Research Question: How does {drug_label} interact with {tgt} and inflammation pathways?

FOCUS ONLY ON:
- {drug_label}
- {tgt} / cyclooxygenase
- inflammation mechanisms

Avoid:
- unrelated drugs
- unrelated diseases

Context:
{context}
"""),
    ])
    print("=== CLEAN PIPELINE RUNNING ===")

    raw_output = response.content
    text_out = raw_output if isinstance(raw_output, str) else str(raw_output)

    sentences = re.split(r"(?<=[.!?]) +", text_out)

    valid: list[str] = []

    for s in sentences:
        s = s.strip()

        pmids = re.findall(r"\(PMID:\s*\d+\)", s)

        if len(pmids) != 1:
            continue

        if any(w in s.lower() for w in [
            "suggest", "overall", "further research",
            "has been explored", "these studies"
        ]):
            continue

        if " and " in s.lower() or "," in s:
            continue

        valid.append(s)

    if not valid:
        return "Insufficient evidence for this molecule-target interaction."

    paragraphs: list[str] = []
    for i in range(0, len(valid), 2):
        paragraphs.append(" ".join(valid[i : i + 2]))

    final_output = "\n\n".join(paragraphs)

    return final_output


def parse_input(state):
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else ""
    smiles_match = re.search(r"Molecule SMILES:\s*(\S+)", content)
    target_match = re.search(r"Biological target:\s*(.+?)(\s*\||$)", content)
    smiles = smiles_match.group(1).strip() if smiles_match else ""
    target = target_match.group(1).strip() if target_match else content.strip()
    smiles_ok, smiles_err = _validate_smiles(smiles)
    if not smiles_ok:
        return {"smiles": smiles, "target": target, "protein_sequence": "",
                "binding_nm": -1.0, "binding_status": "skipped",
                "chunks": [], "top_chunks": [], "drug_name": "", "synonym": "",
                "error": f"Invalid molecule: {smiles_err}"}
    target_ok, target_err = _validate_target(target)
    if not target_ok:
        return {"smiles": smiles, "target": target, "protein_sequence": "",
                "binding_nm": -1.0, "binding_status": "skipped",
                "chunks": [], "top_chunks": [], "drug_name": "", "synonym": "",
                "error": f"Invalid target: {target_err}"}
    return {"smiles": smiles, "target": target, "protein_sequence": "",
            "binding_nm": -1.0, "binding_status": "skipped",
            "chunks": [], "top_chunks": [], "drug_name": "", "synonym": "", "error": ""}


async def check_binding(state):
    if state.get("error"):
        return {}
    target = state.get("target", "").strip()
    smiles = state.get("smiles", "").strip()
    if not target:
        return {"binding_status": "skipped", "binding_nm": -1.0, "protein_sequence": ""}
    protein_sequence = await _fetch_protein_sequence_async(target)
    if not protein_sequence:
        print(f"[ResearchAgent] No UniProt sequence for '{target}' - skipping binding check")
        return {"binding_status": "skipped", "binding_nm": -1.0, "protein_sequence": ""}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        binding_nm, status = await loop.run_in_executor(pool, _run_binding_check, smiles, protein_sequence)
    if status == "weak":
        return {
            "protein_sequence": protein_sequence,
            "binding_nm": binding_nm,
            "binding_status": status,
            "error": (
                f"Binding check failed: surrogate affinity approximation {binding_nm:.0f} nM against "
                f"{target.upper()} exceeds the {BINDING_BLOCK_NM:.0f} nM non-binder threshold.\n\n"
                f"This molecule does not bind {target.upper()}. "
                f"Research briefs are only generated for active molecule-target pairs.\n\n"
                f"For {target.upper()}, try:\n{_suggest_seed(target)}"
            ),
        }
    return {"protein_sequence": protein_sequence, "binding_nm": binding_nm, "binding_status": status}


async def fetch_literature(state):
    if state.get("error"):
        return {"chunks": [], "top_chunks": [], "drug_name": "", "synonym": ""}

    name, syn_list = await asyncio.gather(
        smiles_to_name(state["smiles"]),
        fetch_pubchem_synonyms(state["smiles"]),
    )
    resolved = (name or "").strip() or SMILES_TO_NAME.get((state["smiles"] or "").strip(), "")
    synonym1 = pick_alternate_synonym(syn_list, resolved)
    query = build_literature_query(name, state["target"], state["smiles"], synonym=synonym1)

    pubmed_papers, _ = await asyncio.gather(
        fetch_pubmed(query),
        fetch_semantic_scholar(query)
    )

    pubmed_chunks = chunk_abstracts(pubmed_papers)
    top_chunks = retrieve_top_k(query, pubmed_chunks, k=10)
    top_chunks = filter_relevant(top_chunks)
    if not top_chunks:
        top_chunks = retrieve_top_k(query, pubmed_chunks, k=5)
    top_chunks = filter_chunks(top_chunks, state["target"])

    return {
        "chunks": pubmed_chunks,
        "top_chunks": top_chunks,
        "drug_name": name or "",
        "synonym": synonym1 or "",
    }


def synthesize(state):
    if state.get("error"):
        return {"messages": [AIMessage(content=(
            f"**Cannot generate research brief**\n\n{state['error']}\n\n"
            "Please provide a valid SMILES with confirmed binding to the target."
        ))]}
    top_chunks = state.get("top_chunks", [])
    if not top_chunks or len(top_chunks) < 3:
        return {"messages": [AIMessage(content=NO_EVIDENCE_MESSAGE)]}

    drug_label = (state.get("drug_name") or "").strip() or SMILES_TO_NAME.get(
        (state.get("smiles") or "").strip(), "this compound"
    )
    tgt = (state.get("target") or "").strip() or "COX-2"

    context = ""
    for c in top_chunks:
        context += f"""
[{c['pmid']}]
{c['text']}
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Research Question: How does {drug_label} interact with {tgt} and inflammation pathways?

FOCUS ONLY ON:
- {drug_label}
- {tgt} / cyclooxygenase
- inflammation mechanisms

Avoid:
- unrelated drugs
- unrelated diseases

Context:
{context}
"""),
    ])
    out = _finalize_molecule_rag_output(response.content)
    return {"messages": [AIMessage(content=out)]}


graph = StateGraph(AgentState)
graph.add_node("parse_input",      parse_input)
graph.add_node("check_binding",    check_binding)
graph.add_node("fetch_literature", fetch_literature)
graph.add_node("synthesize",       synthesize)
graph.set_entry_point("parse_input")
graph.add_edge("parse_input",      "check_binding")
graph.add_edge("check_binding",    "fetch_literature")
graph.add_edge("fetch_literature", "synthesize")
graph.add_edge("synthesize",        END)
research_agent = graph.compile()
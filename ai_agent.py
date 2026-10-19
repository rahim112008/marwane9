"""
Agent génomique gratuit — Gemini + Groq avec fallback automatique.
7 outils spécialisés pour la génomique bovine.
"""
import os
import io
import json
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# ═══════════════════════════════════════════════════════════════
# OUTILS (tools) — chacun devient accessible à l'agent
# ═══════════════════════════════════════════════════════════════

@tool
def chercher_pubmed(requete: str) -> str:
    """Cherche des articles scientifiques sur PubMed.
    Arguments : requete = mots-clés (ex: 'DGAT1 bovine milk').
    Retourne les 5 articles les plus pertinents (titre + PMID + date)."""
    try:
        q = urllib.parse.quote(requete)
        url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
               f"?db=pubmed&term={q}&retmode=json&retmax=5&sort=relevance")
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode())
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"Aucun article PubMed pour « {requete} »."
        url2 = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&id={','.join(ids)}&retmode=json")
        with urllib.request.urlopen(url2, timeout=15) as r2:
            d2 = json.loads(r2.read().decode())
        lines = []
        for pid in ids:
            item = d2.get("result", {}).get(pid, {})
            title = item.get("title", "Titre inconnu")
            date = item.get("pubdate", "Date inconnue")
            lines.append(f"- PMID {pid} ({date}) : {title}")
        return f"Articles PubMed pour « {requete} » :\n" + "\n".join(lines)
    except Exception as e:
        return f"Erreur PubMed : {e}"

@tool
def interpreter_fst(fst_valeur: float) -> str:
    """Interprète une valeur FST selon les seuils de Wright.
    Argument : fst_valeur = nombre entre 0 et 1."""
    if fst_valeur < 0.05:
        return f"FST = {fst_valeur} → **différenciation faible** (flux génique important entre populations)."
    elif fst_valeur < 0.15:
        return f"FST = {fst_valeur} → **différenciation modérée** (races distinctes, échanges historiques)."
    elif fst_valeur < 0.25:
        return f"FST = {fst_valeur} → **différenciation forte** (isolement génétique marqué)."
    else:
        return f"FST = {fst_valeur} → **différenciation très forte** (quasi-espèces distinctes)."

@tool
def interpreter_lambda_gc(lambda_val: float) -> str:
    """Interprète λGC d'un GWAS (inflation des p-values).
    Argument : lambda_val = valeur lambda (ex: 1.05)."""
    if lambda_val < 1.05:
        return f"λGC = {lambda_val:.3f} → **excellent** contrôle de structure (attendu ≈ 1.0)."
    elif lambda_val < 1.10:
        return f"λGC = {lambda_val:.3f} → **acceptable** (inflation faible, modèle correct)."
    elif lambda_val < 1.30:
        return f"λGC = {lambda_val:.3f} → **inflation modérée** (structure résiduelle, vérifier PCs et K)."
    else:
        return f"λGC = {lambda_val:.3f} → **inflation forte** (modèle mal calibré, résultats non fiables)."

@tool
def interpreter_heritabilite(h2: float, pheno: str) -> str:
    """Interprète une héritabilité h² selon le phénotype bovin.
    Arguments : h2 (0-1), pheno (ex: 'lait', 'croissance', 'fertilité')."""
    refs = {
        "lait": (0.25, 0.40), "yield": (0.25, 0.40),
        "croissance": (0.30, 0.50), "poids": (0.30, 0.50),
        "fertilité": (0.02, 0.15), "afc": (0.02, 0.15),
        "mamite": (0.05, 0.15), "santé": (0.05, 0.20),
    }
    key = next((k for k in refs if k in pheno.lower()), None)
    if key:
        lo, hi = refs[key]
        if h2 < lo:
            return f"h² = {h2:.2f} pour {pheno} → **faible** (fourchette attendue {lo}-{hi}). Précision de sélection limitée."
        elif h2 <= hi:
            return f"h² = {h2:.2f} pour {pheno} → **normal** (fourchette attendue {lo}-{hi}). Modèle cohérent."
        else:
            return f"h² = {h2:.2f} pour {pheno} → **élevée** (fourchette attendue {lo}-{hi}). Vérifier absence de biais."
    return f"h² = {h2:.2f} pour {pheno} → interprétation dépend du trait (fourchette non connue)."

@tool
def resumer_top_snps(csv_text: str) -> str:
    """Résume les top SNPs d'un GWAS au format CSV avec colonnes SNP,CHR,BP,P,BETA.
    Argument : csv_text = contenu brut du CSV (string)."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        if df.empty:
            return "Aucun SNP fourni."
        df = df.sort_values("P").head(10)
        lines = []
        for _, r in df.iterrows():
            lines.append(f"- {r['SNP']} (BTA{r['CHR']}:{int(r['BP']):,}) "
                         f"P={r['P']:.2e}, β={r['BETA']:+.3f}")
        return "Top 10 SNPs :\n" + "\n".join(lines)
    except Exception as e:
        return f"Erreur lecture CSV : {e}"

@tool
def statistiques_rapides(serie: str) -> str:
    """Statistiques descriptives d'une série de nombres séparés par des virgules.
    Argument : serie (ex: '1.2,3.4,5.6,2.8')."""
    try:
        vals = np.array([float(x) for x in serie.split(",") if x.strip()])
        return (f"n={len(vals)} | moyenne={vals.mean():.4f} | "
                f"médiane={np.median(vals):.4f} | "
                f"écart-type={vals.std():.4f} | "
                f"min={vals.min():.4f} | max={vals.max():.4f}")
    except Exception as e:
        return f"Erreur : {e}"

@tool
def proposer_gene_candidat(region: str) -> str:
    """Propose des gènes candidats connus pour une région bovine (BTA).
    Argument : region = description (ex: 'BTA14 20-30Mb', 'BTA6 35-40Mb')."""
    base = {
        "bta14": "DGAT1 (taux butyreux, locus majeur), CYP11B1 (fertilité), "
                 "HSD17B7 (stéroïdes), TG (lipides)",
        "bta6":  "ABCG2 (protéines lait, mamites), LAP3 (lait), "
                 "SPP1 (immunité), POU1F1 (croissance)",
        "bta20": "GHR (croissance, lait), PRLR (lait), "
                 "NCAPG (croissance), LCORL (taille)",
        "bta2":  "MSTN (musculature, race à viande), IGFBP2 (croissance)",
        "bta5":  "MHC (immunité), BOLA-DRB3 (résistance)",
        "bta18": "PIGR (immunité, mamites)",
        "bta29": "Prion (PRNP, résistance ESB)",
    }
    region_low = region.lower()
    for key, genes in base.items():
        if key in region_low:
            return f"Gènes candidats connus pour {region} :\n{genes}"
    return (f"Aucun gène pré-référencé pour « {region} ». "
            f"Utilise chercher_pubmed pour trouver la littérature.")

# ═══════════════════════════════════════════════════════════════
# FOURNISSEURS LLM
# ═══════════════════════════════════════════════════════════════

def _get_key(name):
    """Cherche la clé dans st.secrets puis dans os.environ."""
    try:
        import streamlit as st
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name)

def _make_gemini(model="gemini-2.5-flash"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = _get_key("GOOGLE_API_KEY") or _get_key("GEMINI_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY absente (aistudio.google.com/apikey)")
    return ChatGoogleGenerativeAI(
        model=model, temperature=0.2, google_api_key=key,
        convert_system_message_to_human=True
    )

def _make_groq(model="openai/gpt-oss-120b"):
    from langchain_groq import ChatGroq
    key = _get_key("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY absente (console.groq.com/keys)")
    return ChatGroq(model=model, temperature=0.2, api_key=key)

# ═══════════════════════════════════════════════════════════════
# CONSTRUCTION DE L'AGENT
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    chercher_pubmed, interpreter_fst, interpreter_lambda_gc,
    interpreter_heritabilite, resumer_top_snps,
    statistiques_rapides, proposer_gene_candidat,
]

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un généticien expert en génomique bovine (races laitières, "
     "allaitantes, zébus). Tu as accès à des outils. Utilise-les "
     "SYSTÉMATIQUEMENT quand la question le justifie :\n"
     "- PubMed : si l'utilisateur demande un gène, une région ou 'que dit la littérature'\n"
     "- interpreter_fst : si on te donne une valeur FST\n"
     "- interpreter_lambda_gc : si on mentionne λGC ou inflation\n"
     "- interpreter_heritabilite : si on donne h² + phénotype\n"
     "- resumer_top_snps : si on te colle un CSV de GWAS\n"
     "- statistiques_rapides : si on te donne une liste de nombres\n"
     "- proposer_gene_candidat : si on mentionne une région BTA\n\n"
     "Réponds en français, structuré, concis (<300 mots). "
     "Cite les PMIDs quand tu utilises PubMed. "
     "Si tu n'as pas besoin d'un outil, réponds directement sans en appeler."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

PROVIDERS = {
    "gemini": {"default_model": "gemini-2.5-flash", "factory": _make_gemini},
    "groq":   {"default_model": "openai/gpt-oss-120b", "factory": _make_groq},
}

def build_agent(provider="gemini", model=None):
    """Retourne (executor, error)."""
    if provider not in PROVIDERS:
        return None, f"Provider inconnu : {provider}"
    info = PROVIDERS[provider]
    try:
        llm = info["factory"](model or info["default_model"])
    except Exception as e:
        return None, f"⚠️ {provider} : {e}"
    agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=TOOLS, verbose=False,
        max_iterations=6, handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor, None

def ask_agent(question, provider="gemini", model=None, fallback=True):
    """
    Envoie une question. Retourne dict :
    {output, provider_used, steps, error}
    """
    order = [provider]
    if fallback:
        order.append("groq" if provider == "gemini" else "gemini")

    last_err = None
    for p in order:
        executor, err = build_agent(p, model if p == provider else None)
        if err:
            last_err = err
            continue
        try:
            r = executor.invoke({"input": question})
            return {
                "output": r.get("output", "Pas de réponse."),
                "provider_used": p,
                "steps": r.get("intermediate_steps", []),
                "error": None,
            }
        except Exception as e:
            last_err = f"⚠️ {p} : {e}"
    return {"output": last_err or "⚠️ Aucun provider disponible.",
            "provider_used": None, "steps": [], "error": last_err}

def available_providers():
    """Retourne la liste des providers configurés (clé présente)."""
    out = []
    if _get_key("GOOGLE_API_KEY") or _get_key("GEMINI_API_KEY"):
        out.append("gemini")
    if _get_key("GROQ_API_KEY"):
        out.append("groq")
    return out

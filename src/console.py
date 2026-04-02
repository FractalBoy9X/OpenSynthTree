"""
Konsolowy runner MolT5 + AiZynthFinder — interaktywny (Docker)

Użycie:
    python console.py

Możliwe polecenia w pętli:
    <nazwa cząsteczki>   — opis przez MolT5 (PubChem lookup)
    smiles <SMILES>      — opis bezpośrednio z SMILES
    caption <opis>       — SMILES z opisu tekstowego
    retro <nazwa/SMILES> — retrosynteza przez AiZynthFinder
    help                 — lista poleceń
    quit / exit / q      — wyjście
"""

import os
import sys
import re
import time
import base64
import subprocess

# ─── Kolory ANSI ──────────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
G   = "\033[92m"
RED = "\033[91m"
Y   = "\033[93m"
C   = "\033[96m"
DIM = "\033[2m"

def ok(m):   print(f"{G}✓ {m}{R}")
def err(m):  print(f"{RED}✗ {m}{R}")
def warn(m): print(f"{Y}⚠  {m}{R}")
def info(m): print(f"{C}{m}{R}")
def dim(m):  print(f"{DIM}{m}{R}")

# ─── Importy krytyczne ────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    err("Brak 'torch'. Uruchom: pip install torch")
    sys.exit(1)

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError:
    err("Brak 'transformers'. Uruchom: pip install transformers")
    sys.exit(1)

try:
    import urllib.request
    import json as _json
    URLLIB_OK = True
except ImportError:
    URLLIB_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    warn("Brak rdkit — wizualizacja niedostępna.")

try:
    from PIL import Image  # noqa
    PILLOW_OK = True
except ImportError:
    PILLOW_OK = False

# ─── Import AiZynthFinder (opcjonalny) ───────────────────────────────────────
try:
    from aizynthfinder.aizynthfinder import AiZynthFinder as _AZF
    AIZYNTH_OK = True
except ImportError:
    AIZYNTH_OK = False
    warn("Brak 'aizynthfinder' — polecenie 'retro' niedostępne.")

# ─── Import rxn-insight (opcjonalny — analiza warunków reakcji) ──────────────
try:
    import io as _io, contextlib as _ctxlib
    with _ctxlib.redirect_stdout(_io.StringIO()), _ctxlib.redirect_stderr(_io.StringIO()):
        from rxn_insight.reaction import Reaction as _RxnInsightReaction
    import pandas as _pd
    RXN_INSIGHT_OK = True
except ImportError:
    RXN_INSIGHT_OK = False

# ─── Konfiguracja (ścieżki kontenerowe) ──────────────────────────────────────
# W Dockerze wymuszamy CPU (brak MPS/CUDA w standardowym obrazie)
DEVICE = "cpu"
if os.environ.get("RETROCHEM_DEVICE"):
    DEVICE = os.environ["RETROCHEM_DEVICE"]

MODEL_S2C       = "laituan245/molt5-large-smiles2caption"
MODEL_C2S       = "laituan245/molt5-large-caption2smiles"

# Ścieżki — nadpisywane przez zmienne środowiskowe (docker-compose)
APP_DIR         = os.environ.get("RETROCHEM_APP_DIR", "/app")
DATA_DIR        = os.environ.get("RETROCHEM_DATA_DIR",
                                 os.path.join(APP_DIR, "data"))
OUTPUT_DIR      = os.environ.get("RETROCHEM_OUTPUT_DIR",
                                 os.path.join(APP_DIR, "output"))
AIZYNTH_CONFIG  = os.path.join(DATA_DIR, "aizynth_data", "config.yml")
RXN_DB_PATH     = os.path.join(DATA_DIR, "rxn_insight_data",
                               "uspto_rxn_insight.gzip")

# ─── Globalny stan (modele ładujemy leniwie) ──────────────────────────────────
_s2c_tok = None
_s2c_mdl = None
_c2s_tok = None
_c2s_mdl = None
_rxn_db  = None   # DataFrame z bazą USPTO dla rxn-insight


def _load_s2c():
    """Ładuje model smiles2caption jeśli jeszcze nie załadowany."""
    global _s2c_tok, _s2c_mdl
    if _s2c_tok is None:
        info("⏳ Ładowanie smiles2caption (pierwsze użycie)...")
        t0 = time.time()
        try:
            _s2c_tok = T5Tokenizer.from_pretrained(MODEL_S2C, model_max_length=512, legacy=False)
            _s2c_mdl = T5ForConditionalGeneration.from_pretrained(MODEL_S2C)
            _s2c_mdl = _s2c_mdl.to(DEVICE)
            _s2c_mdl.eval()
            ok(f"smiles2caption gotowy ({time.time()-t0:.1f}s)")
        except Exception as e:
            err(f"Błąd ładowania modelu: {e}")
            _s2c_tok = _s2c_mdl = None
    return _s2c_tok, _s2c_mdl


def _load_c2s():
    """Ładuje model caption2smiles jeśli jeszcze nie załadowany."""
    global _c2s_tok, _c2s_mdl
    if _c2s_tok is None:
        info("⏳ Ładowanie caption2smiles (pierwsze użycie)...")
        t0 = time.time()
        try:
            _c2s_tok = T5Tokenizer.from_pretrained(MODEL_C2S, model_max_length=512, legacy=False)
            _c2s_mdl = T5ForConditionalGeneration.from_pretrained(MODEL_C2S)
            _c2s_mdl = _c2s_mdl.to(DEVICE)
            _c2s_mdl.eval()
            ok(f"caption2smiles gotowy ({time.time()-t0:.1f}s)")
        except Exception as e:
            err(f"Błąd ładowania modelu: {e}")
            _c2s_tok = _c2s_mdl = None
    return _c2s_tok, _c2s_mdl


# ─── PubChem lookup ───────────────────────────────────────────────────────────
def pubchem_name_to_smiles(name: str) -> str | None:
    """
    Pobiera SMILES z PubChem po nazwie (angielskiej lub systematycznej).
    Używa publicznego REST API — nie wymaga klucza.
    """
    if not URLLIB_OK:
        err("urllib niedostępny — nie można odpytać PubChem.")
        return None
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(name)
        + "/property/IsomericSMILES/JSON"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MolT5-Console/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("SMILES")
        return smiles
    except urllib.error.HTTPError as e:
        if e.code == 404:
            err(f"PubChem nie znalazł cząsteczki: '{name}'")
        else:
            err(f"Błąd HTTP PubChem: {e.code}")
        return None
    except Exception as e:
        err(f"Błąd połączenia z PubChem: {e}")
        return None


import urllib.parse  # noqa: E402 — musi być po definicji, ale działa


# ─── PubChem reverse lookup: SMILES → nazwa ─────────────────────────────────
_name_cache: dict[str, str | None] = {}   # cache w sesji, żeby nie pytać 2x


def pubchem_smiles_to_name(smiles: str) -> str | None:
    """
    Odwrotne wyszukiwanie: SMILES → nazwa zwyczajowa z PubChem.
    Zwraca 'Title' (np. 'aspirin') lub None.
    """
    if smiles in _name_cache:
        return _name_cache[smiles]
    if not URLLIB_OK:
        return None
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        + urllib.parse.quote(smiles, safe="")
        + "/property/IUPACName,Title/JSON"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MolT5-Console/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        name = props.get("Title") or props.get("IUPACName")
        _name_cache[smiles] = name
        return name
    except Exception:
        _name_cache[smiles] = None
        return None


# ─── Wyświetlanie PNG w terminalu ─────────────────────────────────────────────
def _detect_terminal() -> str:
    """
    Wykrywa obsługę inline images w terminalu.
    Zwraca: 'iterm2' | 'kitty' | 'sixel' | 'none'
    """
    term_program = os.environ.get("TERM_PROGRAM", "")
    term         = os.environ.get("TERM", "")
    # VS Code, iTerm2, WezTerm obsługują protokół iTerm2
    if term_program in ("iTerm.app", "WezTerm") or "iterm" in term.lower():
        return "iterm2"
    if term_program == "vscode":
        return "iterm2"
    # Kitty
    if term == "xterm-kitty" or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    return "none"


TERMINAL_KIND = _detect_terminal()


def show_image_inline(path: str, width: int = 60):
    """
    Wyświetla PNG bezpośrednio w terminalu (jeśli obsługiwany).
    width: szerokość w kolumnach znakowych (tylko iTerm2/VSCode).
    W Dockerze zazwyczaj fallback do informacji o ścieżce pliku.
    """
    if not os.path.isfile(path):
        warn(f"Plik nie istnieje: {path}")
        return

    if TERMINAL_KIND == "iterm2":
        _show_iterm2(path, width)
    elif TERMINAL_KIND == "kitty":
        _show_kitty(path)
    else:
        _show_fallback(path)


def _show_iterm2(path: str, width: int):
    """Protokół inline iTerm2 — działa w iTerm2, VS Code terminal, WezTerm."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    name_b64 = base64.b64encode(os.path.basename(path).encode()).decode("ascii")
    size = len(data)
    seq = (
        f"\033]1337;File=name={name_b64};size={size};inline=1;width={width}:"
        + b64
        + "\a"
    )
    sys.stdout.write(seq)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _show_kitty(path: str):
    """Protokół Kitty graphics — uruchamia icat jeśli dostępny."""
    try:
        subprocess.run(["kitty", "+kitten", "icat", "--align", "left", path], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _show_fallback(path)


def _show_fallback(path: str):
    """Fallback: informacja o ścieżce pliku (kontener Docker)."""
    abspath = os.path.abspath(path)
    info(f"Obraz zapisany: {abspath}")
    info("(zamontuj volume output/ aby uzyskać dostęp z hosta)")


# ─── Generowanie opisu (SMILES → tekst) ──────────────────────────────────────
def describe_smiles(smiles: str) -> str | None:
    """Generuje opis tekstowy dla podanego SMILES."""
    tok, mdl = _load_s2c()
    if tok is None:
        return None
    try:
        inputs = tok(
            smiles, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        err(f"Błąd generowania opisu: {e}")
        return None


# ─── Generowanie SMILES (tekst → SMILES) ─────────────────────────────────────
def generate_smiles(caption: str) -> str | None:
    """Generuje SMILES z opisu tekstowego."""
    tok, mdl = _load_c2s()
    if tok is None:
        return None
    try:
        inputs = tok(
            caption, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
        raw = tok.decode(out[0], skip_special_tokens=True)
        return re.sub(r"\s+", "", raw).strip()
    except Exception as e:
        err(f"Błąd generowania SMILES: {e}")
        return None


# ─── Walidacja i wizualizacja ─────────────────────────────────────────────────
def validate_smiles(smiles: str) -> bool:
    if not RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def save_image(smiles: str, label: str) -> str | None:
    """Zapisuje PNG struktury 2D do output/. Zwraca ścieżkę lub None."""
    if not RDKIT_OK or not PILLOW_OK:
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = re.sub(r'[^\w\-_. ]', '_', label).replace(' ', '_')
    path = os.path.join(OUTPUT_DIR, f"{safe}.png")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(400, 300))
        img.save(path)
        return path
    except Exception as e:
        err(f"Błąd zapisu PNG: {e}")
        return None


# ─── AiZynthFinder — retrosynteza ────────────────────────────────────────────
_azf_finder = None   # instancja załadowana leniwie (trwa ~30s)


def _load_finder():
    """Ładuje AiZynthFinder z modelami USPTO (tylko raz na sesję)."""
    global _azf_finder
    if _azf_finder is not None:
        return _azf_finder
    if not AIZYNTH_OK:
        err("AiZynthFinder niedostępny.")
        return None
    if not os.path.isfile(AIZYNTH_CONFIG):
        err(f"Brak pliku konfiguracji: {AIZYNTH_CONFIG}")
        err("Uruchom: docker compose run --rm app download")
        return None
    info("⏳ Ładowanie AiZynthFinder (pierwsze użycie, ~30s)...")
    t0 = time.time()
    try:
        finder = _AZF(configfile=AIZYNTH_CONFIG)
        finder.expansion_policy.select(["uspto", "ringbreaker"])
        finder.filter_policy.select("uspto")
        if finder.stock.items:
            finder.stock.select(list(finder.stock.items))
        _azf_finder = finder
        ok(f"AiZynthFinder gotowy ({time.time()-t0:.1f}s)")
        return _azf_finder
    except Exception as e:
        err(f"Błąd ładowania AiZynthFinder: {e}")
        return None


# ─── rxn-insight — analiza warunków reakcji ─────────────────────────────────
_CHEM_NAMES = {
    "ClCCl": "DCM", "C(Cl)Cl": "DCM",
    "CCCCCC": "heksan", "c1ccncc1": "pirydyna",
    "CC(C)=O": "aceton", "CC=O": "acetaldehyd",
    "CO": "metanol", "CCO": "etanol",
    "CCCO": "propanol", "CCCCO": "butanol",
    "CCOC(=O)C": "octan etylu", "CC(=O)OCC": "octan etylu",
    "CC(=O)O": "kwas octowy",
    "CS(C)=O": "DMSO", "CN(C)C=O": "DMF",
    "C1CCOC1": "THF", "CCOCC": "eter dietylowy",
    "O": "woda", "CC#N": "acetonitryl",
    "C1COCCO1": "dioksan", "ClC(Cl)Cl": "chloroform",
    "C(=O)(Cl)Cl": "chlorek metylenu",
    "c1ccccc1": "benzen", "Cc1ccccc1": "toluen",
    "CC(C)O": "izopropanol",
    "CCN(CC)CC": "trietyloamina (Et3N)",
    "CN(C)C=1C=CN=CC1": "DMAP", "CN(C1=CC=NC=C1)C": "DMAP",
    "O=C([O-])[O-].[Na+].[Na+]": "Na2CO3",
    "O=C([O-])[O-].[K+].[K+]": "K2CO3",
    "[OH-].[Na+]": "NaOH", "[OH-].[K+]": "KOH",
    "[Pd]": "Pd", "c1ccc(P(c2ccccc2)c2ccccc2)cc1": "PPh3",
}


def _chem_name(smiles: str) -> str:
    """Zamienia SMILES na nazwę (jeśli znana) lub zwraca SMILES."""
    if not smiles:
        return ""
    return _CHEM_NAMES.get(smiles, smiles)


def _load_rxn_db():
    """Ładuje bazę USPTO dla rxn-insight (tylko raz na sesję)."""
    global _rxn_db
    if _rxn_db is not None:
        return _rxn_db
    if not RXN_INSIGHT_OK:
        return None
    if not os.path.isfile(RXN_DB_PATH):
        warn(f"Brak bazy rxn-insight: {RXN_DB_PATH}")
        warn("Warunki reakcji (katalizator, rozpuszczalnik) nie będą dostępne.")
        warn("Uruchom: docker compose run --rm app download")
        return None
    info("⏳ Ładowanie bazy USPTO rxn-insight (pierwsze użycie, ~10s)...")
    t0 = time.time()
    try:
        _rxn_db = _pd.read_parquet(RXN_DB_PATH)
        ok(f"Baza rxn-insight załadowana: {len(_rxn_db)} reakcji ({time.time()-t0:.1f}s)")
        return _rxn_db
    except Exception as e:
        err(f"Błąd ładowania bazy rxn-insight: {e}")
        return None


_rxn_mapper = None  # globalny RXNMapper (reużywany)


def _get_rxn_mapper():
    """Zwraca globalną instancję RXNMapper."""
    global _rxn_mapper
    if _rxn_mapper is None:
        from rxnmapper import RXNMapper
        _rxn_mapper = RXNMapper()
    return _rxn_mapper


def _analyze_reaction(product_smiles: str,
                      reactant_smiles_list: list[str]) -> dict | None:
    """
    Analizuje reakcję przez rxn-insight.
    product_smiles: czysty SMILES produktu (z węzła mol).
    reactant_smiles_list: lista czystych SMILES substratów.
    Zwraca dict z CLASS, NAME, FG_REACTANTS itp. lub None.
    """
    if not RXN_INSIGHT_OK:
        return None
    if not product_smiles or not reactant_smiles_list:
        return None
    reactants = ".".join(reactant_smiles_list)
    forward = f"{reactants}>>{product_smiles}"
    try:
        buf = _io.StringIO()
        with _ctxlib.redirect_stdout(buf), _ctxlib.redirect_stderr(buf):
            rxn = _RxnInsightReaction(forward, rxn_mapper=_get_rxn_mapper())
            info_dict = rxn.get_reaction_info()
        # sugestie warunków z bazy USPTO
        db = _load_rxn_db()
        if db is not None:
            try:
                with _ctxlib.redirect_stdout(_io.StringIO()), _ctxlib.redirect_stderr(_io.StringIO()):
                    cond = rxn.suggest_conditions(db)
                info_dict["_suggested_solvent"] = cond.get("Solvent", "")
                info_dict["_suggested_catalyst"] = cond.get("Catalyst", "")
                info_dict["_suggested_reagent"] = cond.get("Reagent", "")
                # top 3 z rankingów
                sol_df = rxn.suggested_solvent
                cat_df = rxn.suggested_catalyst
                rea_df = rxn.suggested_reagent
                info_dict["_top_solvents"] = list(
                    sol_df["NAME"].head(3)) if sol_df is not None and len(sol_df) else []
                info_dict["_top_catalysts"] = list(
                    cat_df["NAME"].head(3)) if cat_df is not None and len(cat_df) else []
                info_dict["_top_reagents"] = list(
                    rea_df["NAME"].head(3)) if rea_df is not None and len(rea_df) else []
            except Exception:
                pass
        return info_dict
    except Exception as e:
        dim(f"  rxn-insight: pominięto reakcję ({e})")
        return None


def _enrich_tree_with_conditions(node: dict,
                                 _parent_mol: str = "") -> None:
    """Wzbogaca węzły reakcji w drzewie o dane z rxn-insight (in-place)."""
    if not RXN_INSIGHT_OK:
        return

    cur_mol = _parent_mol
    if node.get("type") == "mol":
        cur_mol = node.get("smiles", "")

    elif node.get("type") == "reaction":
        reactants = [c.get("smiles", "")
                     for c in node.get("children", [])
                     if c.get("type") == "mol" and c.get("smiles")]
        if _parent_mol and reactants:
            rxn_info = _analyze_reaction(_parent_mol, reactants)
            if rxn_info:
                meta = node.setdefault("metadata", {})
                meta["rxn_class"] = rxn_info.get("CLASS", "")
                meta["rxn_name"] = rxn_info.get("NAME", "")
                meta["rxn_fg_reactants"] = rxn_info.get("FG_REACTANTS", [])
                meta["rxn_fg_products"] = rxn_info.get("FG_PRODUCTS", [])
                meta["rxn_byproducts"] = rxn_info.get("BY-PRODUCTS", [])
                meta["rxn_solvent"] = rxn_info.get("_suggested_solvent", "")
                meta["rxn_catalyst"] = rxn_info.get("_suggested_catalyst", "")
                meta["rxn_reagent"] = rxn_info.get("_suggested_reagent", "")
                meta["rxn_top_solvents"] = rxn_info.get("_top_solvents", [])
                meta["rxn_top_catalysts"] = rxn_info.get("_top_catalysts", [])
                meta["rxn_top_reagents"] = rxn_info.get("_top_reagents", [])

    for child in node.get("children", []):
        _enrich_tree_with_conditions(child, cur_mol)


def _collect_reaction_nodes(node: dict, result: list):
    """Rekurencyjnie zbiera węzły reakcji z drzewa trasy."""
    if node.get("type") == "reaction":
        result.append(node)
    for child in node.get("children", []):
        _collect_reaction_nodes(child, result)


def _parse_route_tree(node: dict, indent: int = 0) -> list[str]:
    """
    Rekurencyjnie przetwarza drzewo retrosytezy (format dict z to_dict()).
    Zwraca listę linii tekstowych gotowych do wydrukowania.
    """
    lines = []
    pad = "  " * indent
    ntype = node.get("type", "")

    if ntype == "mol":
        smiles   = node.get("smiles", "?")
        in_stock = node.get("in_stock", False)
        stock_tag = f"{G}[w magazynie]{R}" if in_stock else f"{Y}[brak]{R}"
        lines.append(f"{pad}{B}Cząsteczka:{R} {smiles}  {stock_tag}")

    elif ntype == "reaction":
        meta  = node.get("metadata", {})
        prob  = meta.get("policy_probability", 0)
        cls   = meta.get("classification", "–")
        occur = meta.get("library_occurence", "?")
        lines.append(
            f"{pad}{C}Reakcja:{R}  "
            f"prawdopodobieństwo={prob:.3f}  "
            f"klasa={cls}  "
            f"wystąpienia w USPTO={occur}"
        )
        # rxn-insight: typ reakcji i grupy funkcyjne
        rxn_name = meta.get("rxn_name", "")
        rxn_class = meta.get("rxn_class", "")
        if rxn_name and rxn_name != "OtherReaction":
            lines.append(f"{pad}  {B}Typ reakcji:{R} {rxn_name} ({rxn_class})")
        elif rxn_class:
            lines.append(f"{pad}  {B}Klasa reakcji:{R} {rxn_class}")
        fg_r = meta.get("rxn_fg_reactants", [])
        fg_p = meta.get("rxn_fg_products", [])
        if fg_r:
            lines.append(f"{pad}  {DIM}Grupy funkcyjne substratów: {', '.join(fg_r)}{R}")
        if fg_p:
            lines.append(f"{pad}  {DIM}Grupy funkcyjne produktów:  {', '.join(fg_p)}{R}")
        byp = meta.get("rxn_byproducts", [])
        if byp:
            lines.append(f"{pad}  {Y}Produkty uboczne: {', '.join(byp)}{R}")
        # warunki z bazy USPTO
        cat = _chem_name(meta.get("rxn_catalyst", ""))
        sol = _chem_name(meta.get("rxn_solvent", ""))
        rea = _chem_name(meta.get("rxn_reagent", ""))
        if cat or sol or rea:
            cond_parts = []
            if cat:
                cond_parts.append(f"katalizator: {cat}")
            if sol:
                cond_parts.append(f"rozpuszczalnik: {sol}")
            if rea:
                cond_parts.append(f"reagent: {rea}")
            lines.append(f"{pad}  {G}Warunki: {' | '.join(cond_parts)}{R}")
            # top 3 alternatywy
            alt_cat = [_chem_name(c) for c in meta.get("rxn_top_catalysts", []) if c]
            alt_sol = [_chem_name(s) for s in meta.get("rxn_top_solvents", []) if s]
            if len(alt_cat) > 1:
                lines.append(f"{pad}  {DIM}  Katalizatory (top 3): {', '.join(alt_cat)}{R}")
            if len(alt_sol) > 1:
                lines.append(f"{pad}  {DIM}  Rozpuszczalniki (top 3): {', '.join(alt_sol)}{R}")
        # wydrukuj uproszczony SMARTS szablonu
        tmpl = meta.get("template", "")
        if tmpl:
            lines.append(f"{pad}{DIM}  szablon: {tmpl[:80]}{'…' if len(tmpl)>80 else ''}{R}")

    for child in node.get("children", []):
        lines.extend(_parse_route_tree(child, indent + 1))

    return lines


def _collect_mol_nodes(node: dict, result: list):
    """Rekurencyjnie zbiera wszystkie węzły cząsteczek z drzewa trasy."""
    if node.get("type") == "mol":
        result.append(node)
    for child in node.get("children", []):
        _collect_mol_nodes(child, result)


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Łamie tekst na linie o max długości max_chars."""
    words, lines, current = text.split(), [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines if lines else [text]


def _try_load_font(paths: list[str], size: int):
    """Próbuje załadować font z listy ścieżek (macOS + Linux)."""
    from PIL import ImageFont as PilFont
    for p in paths:
        try:
            return PilFont.truetype(p, size)
        except Exception:
            continue
    return PilFont.load_default()


# Ścieżki fontów: macOS + Linux (DejaVu — dostępne w Dockerfile)
_SANS_FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",          # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", # Fedora/RHEL
]
_MONO_FONTS = [
    "/System/Library/Fonts/Courier.ttc",             # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
]


def _mol_card(smiles: str, name: str | None = None,
              mol_size: int = 300) -> "Image.Image":
    """
    Renderuje kartę cząsteczki: nazwa (jeśli znana) + struktura 2D + SMILES.
    """
    from PIL import Image as PilImg, ImageDraw as PilDraw

    PAD      = 16
    TXT_H    = 26

    font_scale = max(1.0, mol_size / 300)
    fnt_name   = _try_load_font(_SANS_FONTS, int(20 * font_scale))
    fnt_smiles = _try_load_font(_MONO_FONTS, int(15 * font_scale))

    TXT_H = int(TXT_H * font_scale)

    # ── nazwa ────────────────────────────────────────────────────────────────
    has_name  = bool(name)
    name_h    = (int(30 * font_scale) + 8) if has_name else 0

    # ── struktura RDKit ──────────────────────────────────────────────────────
    struct_img = None
    if RDKIT_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            struct_img = Draw.MolToImage(mol, size=(mol_size, mol_size))
    if struct_img is None:
        struct_img = PilImg.new("RGB", (mol_size, mol_size), "#f0f0f0")
        d = PilDraw.Draw(struct_img)
        d.text((10, mol_size // 2), "?", fill="#aaaaaa")

    # ── SMILES — złam długie napisy ──────────────────────────────────────────
    chars_per_line = max(20, mol_size // 9)
    smiles_lines   = _wrap_text(smiles, chars_per_line)
    smiles_block_h = len(smiles_lines) * TXT_H + PAD

    # ── karta ────────────────────────────────────────────────────────────────
    card_w = mol_size + 2 * PAD
    card_h = PAD + name_h + mol_size + PAD + smiles_block_h + PAD

    card = PilImg.new("RGB", (card_w, card_h), "white")
    draw = PilDraw.Draw(card)

    # cienka szara ramka
    draw.rectangle([0, 0, card_w - 1, card_h - 1], outline="#cccccc", width=2)

    y = PAD

    # pasek nazwy (ciemnoniebieskie tło)
    if has_name:
        draw.rectangle([2, 2, card_w - 3, y + name_h - 2], fill="#1e3a5f")
        draw.text((PAD, y + 3), name, fill="white", font=fnt_name)
        y += name_h

    # struktura
    card.paste(struct_img, (PAD, y))
    y += mol_size + PAD

    # SMILES monospace
    for line in smiles_lines:
        draw.text((PAD, y), line, fill="#555555", font=fnt_smiles)
        y += TXT_H

    return card


def _save_route_image(rt, label: str, route_idx: int,
                      target_name: str = "", score: float = 0.0,
                      solved: bool = False,
                      tree_dict: dict | None = None) -> str | None:
    """
    Zapisuje bogaty PNG wizualizacji trasy retrosytezy.
    """
    from PIL import Image as PilImg, ImageDraw as PilDraw

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = re.sub(r'[^\w\-_. ]', '_', label).replace(' ', '_')
    path = os.path.join(OUTPUT_DIR, f"retro_{safe}_route{route_idx+1}.png")

    try:
        from rxnutils.routes.image import RouteImageFactory

        # ── fonty ────────────────────────────────────────────────────────────
        fnt_title  = _try_load_font(_SANS_FONTS, 36)
        fnt_sub    = _try_load_font(_SANS_FONTS, 26)
        fnt_legend = _try_load_font(_SANS_FONTS, 24)

        # ── 1. Drzewo trasy — wysoka rozdzielczość ──────────────────────────
        factory = RouteImageFactory(
            rt.to_dict(),
            mol_size=550,
            margin=180,
        )
        tree_img = factory.image

        # ── 2. Karty cząsteczek ──────────────────────────────────────────────
        mol_nodes = []
        _collect_mol_nodes(rt.to_dict(), mol_nodes)

        info("  ⏳ Szukam nazw związków w PubChem...")
        mol_names = {}
        for n in mol_nodes:
            smi = n["smiles"]
            if smi not in mol_names:
                mol_names[smi] = pubchem_smiles_to_name(smi)

        MOL_CARD_SIZE = 420
        CARD_PAD      = 20
        CARDS_PER_ROW = min(len(mol_nodes), 3)

        cards = []
        for n in mol_nodes:
            smi  = n["smiles"]
            c = _mol_card(smi, name=mol_names.get(smi), mol_size=MOL_CARD_SIZE)
            cards.append(c)

        # ── dynamiczna szerokość ─────────────────────────────────────────────
        card_w_actual   = cards[0].width if cards else 0
        cards_row_w     = CARDS_PER_ROW * card_w_actual + (CARDS_PER_ROW + 1) * CARD_PAD
        canvas_w        = max(tree_img.width, cards_row_w, 1200)

        # ── 3. Nagłówek ──────────────────────────────────────────────────────
        HDR_H   = 120
        HDR_PAD = 24

        hdr = PilImg.new("RGB", (canvas_w, HDR_H), "#1e3a5f")
        d   = PilDraw.Draw(hdr)
        solved_sym = "✓ rozwiązana" if solved else "✗ nierozwiązana"
        solved_col = "#7dff7d" if solved else "#ffaa55"
        d.text((HDR_PAD, HDR_PAD),
               f"Trasa {route_idx+1}  |  {target_name or label}",
               fill="white", font=fnt_title)
        d.text((HDR_PAD, HDR_PAD + 50),
               f"Score: {score:.4f}   {solved_sym}",
               fill=solved_col, font=fnt_sub)

        # ── 4. Legenda kart ──────────────────────────────────────────────────
        rows       = [cards[i:i+CARDS_PER_ROW]
                      for i in range(0, len(cards), CARDS_PER_ROW)]
        card_h_max = max((c.height for c in cards), default=0)
        legend_title_h = 44
        legend_h   = (legend_title_h
                      + len(rows) * (card_h_max + CARD_PAD)
                      + CARD_PAD)

        legend = PilImg.new("RGB", (canvas_w, legend_h), "#f5f5f5")
        ld     = PilDraw.Draw(legend)
        ld.text((CARD_PAD, CARD_PAD),
                "Cząsteczki w trasie:", fill="#333333", font=fnt_legend)

        y_off = legend_title_h
        for row in rows:
            row_w = len(row) * card_w_actual + (len(row) - 1) * CARD_PAD
            x_off = max(CARD_PAD, (canvas_w - row_w) // 2)
            for card in row:
                legend.paste(card, (x_off, y_off))
                x_off += card_w_actual + CARD_PAD
            y_off += card_h_max + CARD_PAD

        # ── 5. Panel warunków reakcji (rxn-insight) ──────────────────────────
        cond_img = None
        cond_h = 0
        try:
            if tree_dict:
                rxn_nodes = []
                _collect_reaction_nodes(tree_dict, rxn_nodes)
                rxn_with_info = [
                    n for n in rxn_nodes
                    if n.get("metadata", {}).get("rxn_class")
                ]
                if rxn_with_info:
                    COND_PAD = 20
                    LINE_H = 28
                    fnt_cond_title = _try_load_font(_SANS_FONTS, 24)
                    fnt_cond       = _try_load_font(_SANS_FONTS, 20)

                    cond_lines = []
                    for step_i, rn in enumerate(rxn_with_info, 1):
                        m = rn.get("metadata", {})
                        name = m.get("rxn_name", "")
                        cls = m.get("rxn_class", "")
                        if name and name != "OtherReaction":
                            cond_lines.append(
                                (f"Krok {step_i}:", f"{name} ({cls})", True))
                        elif cls:
                            cond_lines.append(
                                (f"Krok {step_i}:", cls, True))
                        cat = _chem_name(m.get("rxn_catalyst", ""))
                        sol = _chem_name(m.get("rxn_solvent", ""))
                        rea = _chem_name(m.get("rxn_reagent", ""))
                        if cat:
                            cond_lines.append(
                                ("", f"Katalizator: {cat}", False))
                        if sol:
                            cond_lines.append(
                                ("", f"Rozpuszczalnik: {sol}", False))
                        if rea:
                            cond_lines.append(
                                ("", f"Reagent: {rea}", False))
                        fg_r = m.get("rxn_fg_reactants", [])
                        if fg_r:
                            cond_lines.append(
                                ("", f"Grupy funkcyjne: {', '.join(fg_r)}", False))
                        byp = m.get("rxn_byproducts", [])
                        if byp:
                            cond_lines.append(
                                ("", f"Produkty uboczne: {', '.join(byp)}", False))

                    if cond_lines:
                        title_h = 40
                        cond_h = (title_h + COND_PAD
                                  + len(cond_lines) * LINE_H + COND_PAD)
                        cond_img = PilImg.new("RGB", (canvas_w, cond_h), "#f0f4f8")
                        cd = PilDraw.Draw(cond_img)
                        cd.rectangle([0, 0, canvas_w - 1, cond_h - 1],
                                     outline="#b0c4de", width=2)
                        cd.text((COND_PAD, COND_PAD),
                                "Analiza reakcji (rxn-insight):",
                                fill="#1e3a5f", font=fnt_cond_title)
                        cy = title_h + COND_PAD
                        for lbl, txt, bold in cond_lines:
                            line_str = f"{lbl} {txt}" if lbl else f"    {txt}"
                            color = "#1e3a5f" if bold else "#555555"
                            cd.text((COND_PAD, cy), line_str,
                                    fill=color, font=fnt_cond)
                            cy += LINE_H
        except Exception as e:
            warn(f"Nie udało się wygenerować panelu warunków na PNG: {e}")

        # ── 6. Złóż finalny obraz ───────────────────────────────────────────
        tree_x  = max(0, (canvas_w - tree_img.width) // 2)
        total_h = HDR_H + tree_img.height + cond_h + legend_h

        final = PilImg.new("RGB", (canvas_w, total_h), "white")
        final.paste(hdr,      (0,      0))
        final.paste(tree_img, (tree_x, HDR_H))
        if cond_img:
            final.paste(cond_img, (0, HDR_H + tree_img.height))
        final.paste(legend,   (0,      HDR_H + tree_img.height + cond_h))

        final.save(path, dpi=(300, 300))
        return path

    except Exception as e:
        err(f"Błąd zapisu obrazu trasy: {e}")
        try:
            rt.to_image().save(path)
            return path
        except Exception:
            return None


MAX_ROUTES = 10
DEFAULT_ROUTES = 3


def handle_retro(query: str, n_routes: int = DEFAULT_ROUTES):
    """Retrosynteza: nazwa lub SMILES → AiZynthFinder → wyświetl trasy."""
    print()
    n_routes = max(1, min(n_routes, MAX_ROUTES))

    if re.search(r'[=#@\[\]\\/()\d]', query) or query.isupper():
        smiles = query.strip()
        label  = smiles[:20]
        info(f"Retrosynteza SMILES: {smiles}")
    else:
        info(f"Szukam '{query}' w PubChem...")
        smiles = pubchem_name_to_smiles(query)
        label  = query
        if smiles is None:
            warn("Spróbuj podać SMILES bezpośrednio: retro <SMILES>")
            return
        ok(f"SMILES: {smiles}")

    finder = _load_finder()
    if finder is None:
        return

    info(f"⏳ Przeszukuję drzewo retrosytezy (maks. 100 iteracji MCTS)...")
    t0 = time.time()
    try:
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
    except Exception as e:
        err(f"Błąd wyszukiwania: {e}")
        return

    elapsed = time.time() - t0
    routes  = finder.routes
    stats   = finder.extract_statistics()
    n_all   = stats.get("number_of_routes", 0)
    n_solved = stats.get("number_of_solved_routes", 0)
    solved  = stats.get("is_solved", False)

    print()
    solved_str = f"{G}TAK{R}" if solved else f"{RED}NIE{R}"
    print(f"{B}Wynik retrosytezy:{R}  rozwiązana={solved_str}  "
          f"tras={n_all}  rozwiązanych={n_solved}  "
          f"czas={elapsed:.1f}s")

    if stats.get("precursors_in_stock"):
        print(f"{B}Prekursory (dostępne):{R}")
        for p in stats["precursors_in_stock"].split(","):
            print(f"  • {p.strip()}")
    if stats.get("precursors_not_in_stock"):
        print(f"{Y}Prekursory (niedostępne):{R}")
        for p in stats["precursors_not_in_stock"].split(","):
            print(f"  • {p.strip()}")

    useful_routes = [
        r for r in routes
        if r["reaction_tree"].to_dict().get("children")
    ]
    if not useful_routes and routes:
        warn("Żadna trasa nie zawiera reakcji — wyświetlam najlepszą.")
        useful_routes = routes[:1]

    show_n = min(n_routes, len(useful_routes))
    print(f"\n{B}Top {show_n} tras (ze {len(routes)} znalezionych, "
          f"{len(routes) - len(useful_routes)} bez reakcji):{R}")

    for i in range(show_n):
        route   = useful_routes[i]
        rt      = route["reaction_tree"]
        score   = route["score"].get("state score", 0)
        solved_r = route["route_metadata"].get("is_solved", False)
        tag     = f"{G}✓ rozwiązana{R}" if solved_r else f"{Y}nierozwiązana{R}"

        print(f"\n{B}━━━ Trasa {i+1}  score={score:.4f}  {tag} ━━━{R}")
        tree_dict = rt.to_dict()
        if RXN_INSIGHT_OK:
            info("  ⏳ Analiza warunków reakcji (rxn-insight)...")
            _enrich_tree_with_conditions(tree_dict)
        for line in _parse_route_tree(tree_dict):
            print(line)

        path = _save_route_image(rt, label, i,
                                 target_name=query, score=score, solved=solved_r,
                                 tree_dict=tree_dict)
        if path:
            show_image_inline(path)

    ok(f"Retrosynteza zakończona. Obrazy w: {os.path.abspath(OUTPUT_DIR)}/")


# ─── Obsługa poleceń ─────────────────────────────────────────────────────────
def handle_name(name: str):
    """Nazwa cząsteczki → PubChem SMILES → MolT5 opis."""
    print()
    info(f"Szukam '{name}' w PubChem...")
    smiles = pubchem_name_to_smiles(name)
    if smiles is None:
        warn("Spróbuj podać nazwę po angielsku lub bezpośrednio SMILES: smiles <SMILES>")
        return

    ok(f"SMILES: {smiles}")

    valid = validate_smiles(smiles)
    if valid is False:
        warn("RDKit zgłasza niepoprawny SMILES — mimo to próbuję opis.")

    info("⏳ Generowanie opisu przez MolT5...")
    t0 = time.time()
    caption = describe_smiles(smiles)
    if caption:
        print()
        print(f"{B}Opis MolT5:{R}")
        print(f"  {caption}")
        print(f"{DIM}  (czas: {time.time()-t0:.1f}s){R}")
    else:
        err("Nie udało się wygenerować opisu.")

    path = save_image(smiles, name)
    if path:
        show_image_inline(path)


def handle_smiles_cmd(smiles: str):
    """SMILES → MolT5 opis."""
    print()
    valid = validate_smiles(smiles)
    if valid is False:
        warn(f"RDKit: SMILES może być niepoprawny — próbuję mimo to.")
    elif valid is True:
        ok("RDKit: SMILES poprawny ✓")

    info("⏳ Generowanie opisu przez MolT5...")
    t0 = time.time()
    caption = describe_smiles(smiles)
    if caption:
        print()
        print(f"{B}Opis MolT5:{R}")
        print(f"  {caption}")
        print(f"{DIM}  (czas: {time.time()-t0:.1f}s){R}")
    else:
        err("Nie udało się wygenerować opisu.")

    label = smiles[:20].replace("/", "").replace("\\", "")
    path = save_image(smiles, label)
    if path:
        show_image_inline(path)


def handle_caption_cmd(caption: str):
    """Opis tekstowy → MolT5 SMILES."""
    print()
    info("⏳ Generowanie SMILES przez MolT5...")
    t0 = time.time()
    smiles = generate_smiles(caption)
    if smiles is None:
        err("Nie udało się wygenerować SMILES.")
        return

    print()
    print(f"{B}Wygenerowany SMILES:{R}")
    print(f"  {smiles}")
    print(f"{DIM}  (czas: {time.time()-t0:.1f}s){R}")

    valid = validate_smiles(smiles)
    if valid is True:
        ok("RDKit: SMILES chemicznie poprawny ✓")
    elif valid is False:
        err("RDKit: SMILES niepoprawny ✗")

    path = save_image(smiles, caption[:30])
    if path:
        show_image_inline(path)


def print_help():
    print(f"""
{B}Dostępne polecenia:{R}

  {C}<nazwa>{R}               Nazwa cząsteczki → PubChem SMILES → opis MolT5
                        Przykłady: aspirin, caffeine, ibuprofen, ethanol
                        (nazwy po angielsku działają najlepiej)

  {C}smiles <SMILES>{R}       Bezpośredni SMILES → opis MolT5
                        Przykład: smiles CC(=O)Oc1ccccc1C(=O)O

  {C}caption <opis>{R}        Opis tekstowy → SMILES MolT5
                        Przykład: caption The molecule is a simple aromatic alcohol

  {C}retro <nazwa/SMILES> [N]{R} Retrosynteza przez AiZynthFinder (USPTO)
                        + analiza warunków reakcji (rxn-insight)
                        N = liczba tras do wyświetlenia (domyślnie {DEFAULT_ROUTES}, maks. {MAX_ROUTES})
                        Przykłady: retro aspirin
                                   retro aspirin 5
                                   retro CC(=O)Oc1ccccc1C(=O)O 7
                        Wyniki: drzewo reakcji, klasyfikacja, grupy funkcyjne + PNG

  {C}help{R}                  Ta pomoc

  {C}quit / exit / q{R}       Wyjście
""")


# ─── Pętla główna ─────────────────────────────────────────────────────────────
def main():
    print(f"""
{B}{C}╔══════════════════════════════════════════════════╗
║  retro_chem Console — Docker Open-Source         ║
║  MolT5 + AiZynthFinder + rxn-insight             ║
║  SMILES ↔ opis cząsteczki (PubChem)             ║
╚══════════════════════════════════════════════════╝{R}
Wpisz nazwę cząsteczki, 'help' lub 'quit'.
{DIM}Modele ładują się przy pierwszym użyciu (~35s).{R}
{DIM}Urządzenie: {DEVICE}{R}
""")
    if TERMINAL_KIND == "iterm2":
        ok("Terminal obsługuje inline images (iTerm2/VSCode).")
    elif TERMINAL_KIND == "kitty":
        ok("Terminal obsługuje inline images (Kitty).")
    else:
        info("Obrazy PNG zapisywane do: " + OUTPUT_DIR)
    print()

    while True:
        try:
            raw = input(f"{B}{C}retro_chem>{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            info("Do widzenia!")
            break

        if not raw:
            continue

        low = raw.lower()

        if low in ("quit", "exit", "q"):
            info("Do widzenia!")
            break
        elif low == "help":
            print_help()
        elif low.startswith("smiles "):
            smiles_arg = raw[7:].strip()
            if smiles_arg:
                handle_smiles_cmd(smiles_arg)
            else:
                warn("Podaj SMILES po komendzie, np.: smiles CC(=O)O")
        elif low.startswith("caption "):
            cap_arg = raw[8:].strip()
            if cap_arg:
                handle_caption_cmd(cap_arg)
            else:
                warn("Podaj opis po komendzie, np.: caption The molecule is...")
        elif low.startswith("retro "):
            retro_arg = raw[6:].strip()
            if retro_arg:
                parts = retro_arg.rsplit(None, 1)
                if len(parts) == 2 and parts[1].isdigit():
                    handle_retro(parts[0], int(parts[1]))
                else:
                    handle_retro(retro_arg)
            else:
                warn("Podaj nazwę lub SMILES po komendzie, np.: retro aspirin")
        else:
            handle_name(raw)

        print()


if __name__ == "__main__":
    main()

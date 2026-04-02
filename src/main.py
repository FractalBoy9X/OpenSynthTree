"""
MolT5 Demo — SMILES ↔ Opis cząsteczki (Docker)
Uruchamianie modeli laituan245/molt5-large-* na CPU
"""

import os
import sys
import time
import re

# ─── Kolory ANSI ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def info(msg):   print(f"{CYAN}{msg}{RESET}")
def ok(msg):     print(f"{GREEN}✓ {msg}{RESET}")
def err(msg):    print(f"{RED}✗ {msg}{RESET}")
def warn(msg):   print(f"{YELLOW}⚠  {msg}{RESET}")
def section(msg): print(f"\n{BOLD}{CYAN}{'─'*60}\n{msg}\n{'─'*60}{RESET}")

# ─── Krytyczne importy ────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    err("Brak biblioteki 'torch'. Zainstaluj: pip install torch")
    sys.exit(1)

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError:
    err("Brak biblioteki 'transformers'. Zainstaluj: pip install transformers")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_OK = True
except ImportError:
    warn("Brak biblioteki 'rdkit' — walidacja SMILES i wizualizacja będą pominięte.")
    RDKIT_OK = False

try:
    from PIL import Image  # noqa: F401
    PILLOW_OK = True
except ImportError:
    warn("Brak biblioteki 'Pillow' — zapis obrazów PNG będzie pominięty.")
    PILLOW_OK = False

# ─── Konfiguracja (kontener Docker) ──────────────────────────────────────────
DEVICE = os.environ.get("RETROCHEM_DEVICE", "cpu")
OUTPUT_DIR = os.environ.get("RETROCHEM_OUTPUT_DIR", "/app/output")

# ─── Dane wejściowe ───────────────────────────────────────────────────────────
MOLECULES = [
    {"name": "Aspiryna",    "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"name": "Kofeina",     "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"},
    {"name": "Ibuprofen",   "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
    {"name": "Paracetamol", "smiles": "CC(=O)Nc1ccc(O)cc1"},
    {"name": "Penicylina G","smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"},
]

DESCRIPTIONS = [
    "The molecule is a simple aromatic alcohol also known as phenol.",
    "The molecule contains a carboxylic acid group attached to a benzene ring.",
]

MODEL_SMILES2CAP = "laituan245/molt5-large-smiles2caption"
MODEL_CAP2SMILES = "laituan245/molt5-large-caption2smiles"

# ─── Funkcje ładowania modeli ─────────────────────────────────────────────────
def load_caption_model():
    """Ładuje model SMILES → opis (smiles2caption)."""
    info(f"⏳ Ładowanie modelu {MODEL_SMILES2CAP} ...")
    t0 = time.time()
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_SMILES2CAP,
            model_max_length=512,
        )
        model = T5ForConditionalGeneration.from_pretrained(MODEL_SMILES2CAP)
        model = model.to(DEVICE)
        model.eval()
        elapsed = time.time() - t0
        ok(f"Model smiles2caption załadowany w {elapsed:.1f}s (urządzenie: {DEVICE})")
        return tokenizer, model
    except OSError as e:
        err(f"Błąd ładowania smiles2caption: {e}")
        err("Sprawdź połączenie z internetem lub ścieżkę do modelu.")
        return None, None
    except Exception as e:
        err(f"Nieoczekiwany błąd przy ładowaniu smiles2caption: {e}")
        return None, None


def load_generation_model():
    """Ładuje model opis → SMILES (caption2smiles)."""
    info(f"⏳ Ładowanie modelu {MODEL_CAP2SMILES} ...")
    t0 = time.time()
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_CAP2SMILES,
            model_max_length=512,
        )
        model = T5ForConditionalGeneration.from_pretrained(MODEL_CAP2SMILES)
        model = model.to(DEVICE)
        model.eval()
        elapsed = time.time() - t0
        ok(f"Model caption2smiles załadowany w {elapsed:.1f}s (urządzenie: {DEVICE})")
        return tokenizer, model
    except OSError as e:
        err(f"Błąd ładowania caption2smiles: {e}")
        err("Sprawdź połączenie z internetem lub ścieżkę do modelu.")
        return None, None
    except Exception as e:
        err(f"Nieoczekiwany błąd przy ładowaniu caption2smiles: {e}")
        return None, None


# ─── Zadanie 1: SMILES → opis ─────────────────────────────────────────────────
def caption_molecules(tokenizer, model) -> list[dict]:
    """Generuje opisy tekstowe dla listy cząsteczek SMILES."""
    results = []
    for mol in MOLECULES:
        name   = mol["name"]
        smiles = mol["smiles"]
        info(f"  ⏳ Generowanie opisu dla: {name} ({smiles})")
        try:
            inputs = tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                )

            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ok(f"  {name}: {caption}")
            results.append({"name": name, "smiles": smiles, "caption": caption})
        except Exception as e:
            err(f"  Błąd generowania dla {name}: {e}")
            results.append({"name": name, "smiles": smiles, "caption": None})
    return results


# ─── Zadanie 2: opis → SMILES ─────────────────────────────────────────────────
def validate_smiles(smiles_str: str) -> bool:
    """Waliduje SMILES przez RDKit. Zwraca True jeśli poprawny chemicznie."""
    if not RDKIT_OK:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        return mol is not None
    except Exception:
        return False


def generate_smiles_from_text(tokenizer, model) -> list[dict]:
    """Generuje SMILES na podstawie opisów tekstowych."""
    results = []
    for desc in DESCRIPTIONS:
        info(f"  ⏳ Opis → SMILES: \"{desc[:60]}...\"" if len(desc) > 60 else f"  ⏳ Opis → SMILES: \"{desc}\"")
        try:
            inputs = tokenizer(
                desc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                )

            raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_smiles = re.sub(r"\s+", "", raw).strip()

            valid = validate_smiles(generated_smiles)
            if valid is True:
                ok(f"  SMILES: {generated_smiles}  [poprawny ✓]")
            elif valid is False:
                err(f"  SMILES: {generated_smiles}  [niepoprawny ✗]")
            else:
                warn(f"  SMILES: {generated_smiles}  [walidacja pominięta — brak rdkit]")

            results.append({
                "description": desc,
                "smiles": generated_smiles,
                "valid": valid,
            })
        except Exception as e:
            err(f"  Błąd generowania SMILES: {e}")
            results.append({"description": desc, "smiles": None, "valid": False})
    return results


# ─── Zadanie 3: wizualizacja ──────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    """Usuwa znaki niedozwolone w nazwach plików."""
    return re.sub(r'[^\w\-_. ]', '_', name).replace(' ', '_')


def render_molecule_images(molecules: list[dict]):
    """Renderuje struktury 2D cząsteczek i zapisuje jako PNG do output/."""
    if not RDKIT_OK:
        warn("Pominięcie wizualizacji — brak rdkit.")
        return
    if not PILLOW_OK:
        warn("Pominięcie zapisu PNG — brak Pillow.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for mol_data in molecules:
        name   = mol_data["name"]
        smiles = mol_data["smiles"]
        if not smiles:
            warn(f"  Pominięcie {name} — brak SMILES.")
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                err(f"  Niepoprawny SMILES dla {name}: {smiles}")
                continue
            img = Draw.MolToImage(mol, size=(400, 300))
            filename = sanitize_filename(name) + ".png"
            path = os.path.join(OUTPUT_DIR, filename)
            img.save(path)
            ok(f"  Zapisano: {path}")
        except Exception as e:
            err(f"  Błąd renderowania {name}: {e}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    section("MolT5 Demo — retro_chem Docker")
    info(f"Python: {sys.version}")
    info(f"PyTorch: {torch.__version__}")
    info(f"Urządzenie: {DEVICE}")

    # ── Zadanie 1: SMILES → opis ──────────────────────────────────────────────
    section("Zadanie 1: SMILES → Opis cząsteczki")
    tok1, mdl1 = load_caption_model()
    caption_results = []
    if tok1 and mdl1:
        t1 = time.time()
        caption_results = caption_molecules(tok1, mdl1)
        ok(f"Zadanie 1 ukończone w {time.time() - t1:.1f}s")
        del mdl1, tok1
    else:
        err("Pominięcie zadania 1 — model niedostępny.")

    # ── Zadanie 2: opis → SMILES ──────────────────────────────────────────────
    section("Zadanie 2: Opis → SMILES")
    tok2, mdl2 = load_generation_model()
    if tok2 and mdl2:
        t2 = time.time()
        smiles_results = generate_smiles_from_text(tok2, mdl2)
        ok(f"Zadanie 2 ukończone w {time.time() - t2:.1f}s")
        del mdl2, tok2
    else:
        err("Pominięcie zadania 2 — model niedostępny.")
        smiles_results = []

    # ── Zadanie 3: wizualizacja ───────────────────────────────────────────────
    section("Zadanie 3: Wizualizacja cząsteczek (RDKit)")
    t3 = time.time()
    render_molecule_images(caption_results if caption_results else MOLECULES)
    ok(f"Zadanie 3 ukończone w {time.time() - t3:.1f}s")

    # ── Podsumowanie ──────────────────────────────────────────────────────────
    total = time.time() - t_start
    section(f"Gotowe! Łączny czas: {total:.1f}s")
    info(f"Obrazy PNG zapisane w: {os.path.abspath(OUTPUT_DIR)}/")

    if smiles_results:
        info("\nPodsumowanie zadania 2:")
        for r in smiles_results:
            status = "✓" if r["valid"] else ("✗" if r["valid"] is False else "?")
            print(f"  [{status}] {r['smiles']}  ←  \"{r['description'][:50]}...\"" if len(r['description']) > 50
                  else f"  [{status}] {r['smiles']}  ←  \"{r['description']}\"")


if __name__ == "__main__":
    main()

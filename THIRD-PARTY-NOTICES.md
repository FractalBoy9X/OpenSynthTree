# Third-Party Notices

This project uses the following third-party components. Each component is listed
with its license type and attribution.

---

## PyTorch

- **License:** BSD-3-Clause
- **Copyright:** Copyright (c) 2016-present, Facebook, Inc. (Meta Platforms, Inc.)
- **Source:** https://github.com/pytorch/pytorch
- **Full license:** https://github.com/pytorch/pytorch/blob/main/LICENSE

---

## Hugging Face Transformers

- **License:** Apache License 2.0
- **Copyright:** Copyright 2018- The Hugging Face team. All rights reserved.
- **Source:** https://github.com/huggingface/transformers
- **Full license:** https://github.com/huggingface/transformers/blob/main/LICENSE

---

## MolT5 Pre-trained Models

- **License:** Apache License 2.0 (inherited from T5 / Hugging Face)
- **Models:** `laituan245/molt5-large-smiles2caption`, `laituan245/molt5-large-caption2smiles`
- **Source:** https://huggingface.co/laituan245
- **Paper:** Edwards, C., Lai, T., Ros, K., Honke, G., & Ji, H. (2022).
  *Translation between Molecules and Natural Language.* EMNLP 2022.
  https://arxiv.org/abs/2204.11817

Models are downloaded at runtime from Hugging Face Hub and cached locally.
They are not distributed with this project.

---

## RDKit

- **License:** BSD-3-Clause
- **Copyright:** Copyright (c) 2006-2025, Greg Landrum and other RDKit contributors
- **Source:** https://github.com/rdkit/rdkit
- **Full license:** https://github.com/rdkit/rdkit/blob/master/license.txt

---

## AiZynthFinder

- **License:** MIT License
- **Copyright:** Copyright (c) 2020, AstraZeneca
- **Source:** https://github.com/MolecularAI/aizynthfinder
- **Full license:** https://github.com/MolecularAI/aizynthfinder/blob/master/LICENSE
- **Paper:** Genheden, S., Thakkar, A., Chadimova, V., et al. (2020).
  *AiZynthFinder: a fast, robust and flexible open-source software for
  retrosynthetic planning.* J. Cheminformatics 12, 70.
  https://doi.org/10.1186/s13321-020-00472-1

Pre-trained models (USPTO ONNX files) are provided by the AiZynthFinder project.
Users should download them via `aizynthfinder.tools.download_public_data`.

---

## rxn-insight

- **License:** MIT License
- **Copyright:** Copyright (c) 2024, Maarten R. Dobbelaere
- **Source:** https://github.com/mrodobbe/Rxn-INSIGHT
- **Full license:** https://github.com/mrodobbe/Rxn-INSIGHT/blob/main/LICENSE
- **Paper:** Dobbelaere, M.R., et al. (2024).
  *Rxn-INSIGHT — fast chemical reaction analysis using bond-electron matrices.*
  J. Cheminformatics 16, 37. https://doi.org/10.1186/s13321-024-00834-z

Used for classifying reaction types, identifying functional groups, and predicting
by-products in retrosynthetic routes.

---

## SentencePiece

- **License:** Apache License 2.0
- **Copyright:** Copyright 2016 Google Inc.
- **Source:** https://github.com/google/sentencepiece
- **Full license:** https://github.com/google/sentencepiece/blob/master/LICENSE

---

## Pillow

- **License:** HPND (Historical Permission Notice and Disclaimer)
- **Copyright:** Copyright (c) 1995-2011 by Secret Labs AB; Copyright (c) 1997-2011 by Fredrik Lundh; Copyright (c) 2010-2025 by Jeffrey A. Clark and contributors
- **Source:** https://github.com/python-pillow/Pillow
- **Full license:** https://github.com/python-pillow/Pillow/blob/main/LICENSE

---

## Rich

- **License:** MIT License
- **Copyright:** Copyright (c) 2020 Will McGugan
- **Source:** https://github.com/Textualize/rich

---

## Typer

- **License:** MIT License
- **Copyright:** Copyright (c) 2019 Sebastian Ramirez
- **Source:** https://github.com/fastapi/typer

---

## PubChem API

- **Provider:** National Center for Biotechnology Information (NCBI), U.S. National Library of Medicine
- **Data license:** Public domain (U.S. government work)
- **Source:** https://pubchem.ncbi.nlm.nih.gov/
- **Terms of use:** https://www.ncbi.nlm.nih.gov/home/about/policies/
- **Citation:** Kim, S., et al. (2023). *PubChem 2023 update.* Nucleic Acids Res.
  https://doi.org/10.1093/nar/gkac956

Data retrieved at runtime via PubChem REST API. Not bundled with this project.

---

## USPTO Reaction Data

- **License:** Public domain (U.S. government work, 17 U.S.C. Section 105)
- **Source:** https://www.uspto.gov/
- **Processed dataset:** Lowe, D. (2017). *Chemical reactions from US patents
  (1976-Sep2016).* https://doi.org/10.6084/m9.figshare.5104873.v1
  (Released under CC0 1.0 Universal)

Reaction templates used by AiZynthFinder are derived from this public domain data.

---

## rxn-insight USPTO Database (Zenodo)

- **License:** CC BY 4.0
- **Source:** https://zenodo.org/records/10171745
- **Citation:** Dobbelaere, M.R., et al. (2024). Rxn-INSIGHT — fast chemical reaction
  analysis using bond-electron matrices. J. Cheminformatics 16, 37.

Downloaded at runtime via `download_data.sh`. Not bundled with this project.

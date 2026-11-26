# Notices

contract-key-terms-mcp
Copyright 2026 It Just Gets Better LLC

The source code is licensed under the Apache License, Version 2.0 (see LICENSE).

## The model weights (`src/key_terms_mcp/models/key_terms_model.json.gz`)

The model was trained on CUAD v1 (Contract Understanding Atticus Dataset) by The Atticus Project, Inc.,
https://www.atticusprojectai.org/cuad, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The model file is shared under the same licence, CC BY 4.0,
with this attribution. CUAD is provided as is, without warranties; see the licence and The Atticus Project's
disclaimer (https://www.atticusprojectai.org/legal).

Changes: the model holds no CUAD sentence or passage. It holds a vocabulary of 30,000 lower-case words and
two-word pairs that each appear in at least two CUAD sentences (these can include names and numbers from the
contracts), how common each is across those sentences, and, per clause type, weights learned from which
sentences the dataset's lawyers labelled with that type; the labels for document name, parties and dates were
not used.

## The clause type names (`src/key_terms_mcp/clause_types.py`)

The CUAD category names are from CUAD v1 by The Atticus Project, Inc., CC BY 4.0, as above.

## The sample contracts (`src/key_terms_mcp/samples/`)

Three contracts copied unchanged from CUAD v1, CC BY 4.0, as above. Their source, release and checksums are
in `src/key_terms_mcp/samples/SOURCE.md`.

## Citation (a courtesy, not a licence condition)

Dan Hendrycks, Collin Burns, Anya Chen and Spencer Ball. "CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review." NeurIPS 2021.

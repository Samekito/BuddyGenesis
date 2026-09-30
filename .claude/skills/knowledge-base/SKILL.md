---
name: knowledge-base
description: How to add, replace or debug SOC Buddy's handbook knowledge (knowledge_base/*.docx) and check what the BM25 retriever returns for a question. Use when the bot answers "I don't know" for something that should be in a handbook, or when handbooks change.
---
# SOC Buddy knowledge base

The bot only knows what is in `knowledge_base/*.docx`. The index is rebuilt in memory at every app start — there is no separate build step.

## Add or replace a handbook
1. Drop the `.docx` into `knowledge_base/`. Name it `<DEPT> HANDBOOK<n>.docx` (e.g. `CSC HANDBOOK3.docx`).
2. If `<DEPT>` is a new department code, add its full name to `DEPARTMENT_NAMES` in `buddy/knowledge.py` so searches by department name match.
3. Delete the old version — files with `copy` in the name and Word lock files (`~$...`) are skipped, but any other duplicate would be indexed twice.
4. Restart the app.

## Check what retrieval returns for a question
```bash
.venv/Scripts/python scripts/search.py "how many units is CSC 101"
```
Prints the top chunks with their source and score. If the right chunk is missing, the wording in the question does not overlap the handbook text — BM25 is keyword-based.

## Only .docx is read
`.txt` and `.pdf` are ignored on purpose (the old `.txt` exports were lossy and duplicated the .docx content).

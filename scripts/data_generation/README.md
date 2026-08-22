# Data generation scripts

These regenerate every data asset from scratch. All are **seeded and
reproducible** — rerunning produces byte-identical output.

You do not need to run these. `data/` is already populated. They exist so the
data is reproducible rather than a mystery blob, which matters for the
"reproducibility" claim in the project README.

## Order matters

```bash
cd scripts/data_generation

python build_pdfs.py          # data/_source/*.md  ->  data/documents/*.pdf
python seed_db.py             # ->  data/db/pacify.db
python gen_intents.py         # ->  data/intents/train.csv
python gen_testset.py         # ->  data/intents/test_hard.csv
python gen_tickets.py         # ->  data/tickets/ticket_history.csv
python gen_evalsets.py        # ->  data/eval/*.json
python check_consistency.py   # verify corpus vs canonical_facts.md
```

Then rebuild the database views:

```bash
cd ../..
python scripts/setup_database.py
```

## Editing the corpus

**Never edit the PDFs directly.** Edit `data/_source/*.md` and rerun
`build_pdfs.py`. Section IDs (`S1`, `S2.3`) are referenced by the evaluation
sets and must never be renumbered.

After any corpus edit, run `check_consistency.py`. It verifies 16 canonical
facts, confirms all 18 planted defects are still present, and checks that the
8 deliberate omissions have not leaked in.

## Requirements

```bash
pip install reportlab faker
```

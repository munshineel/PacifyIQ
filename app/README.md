# PacifyIQ — Streamlit interface

```bash
streamlit run app/Home.py
```

## Structure

```
app/
  Home.py                       overview, setup status
  pages/
    1_Support_Agent.py          main interface
    2_Screenshot_Analysis.py    standalone vision
    3_Knowledge_Base.py         search and document browser
    4_Conversation_History.py   logged requests
    5_Analytics.py              simulated + live
    6_Evaluation.py             measured results
```

## Architecture

Pages import **only** `src.ui.service` and `src.ui.components`. They never
import the agent, retriever, guardrails or evaluation framework directly.

```
app/pages/*  ->  src/ui/service.py  ->  src/agent, src/rag, src/knowledge, ...
             ->  src/ui/components.py   (presentation only)
```

`src/` never imports `streamlit`. `tests/test_architecture.py` asserts both
directions.

Every failure mode — missing index, unset API key, corrupt upload, oversized
image, tool error — is handled once in the service layer and returned as a typed
result, so pages never show a traceback.

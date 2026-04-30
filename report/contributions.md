# Team Contributions

**Siddhanth Gouru**
- Data preprocessing pipeline (`scripts/preprocess.py`)
- Baseline replication and evaluation framework (`models/baselines.py`, `scripts/evaluate.py`)
- Manual error annotation and taxonomy design (`scripts/error_taxonomy.py`)

**Sidharth Pasula**
- Negation-aware preprocessing intervention (negation marking via NEGATION_CUES, `scripts/interventions.py`)
- Implicit aspect identification in error taxonomy
- Error taxonomy results (`results/errors.json`, `results/interventions.json`)

**Tanay Shrivastava**
- Conflict-class oversampling intervention (`scripts/interventions.py`)
- Multi-aspect sentence decomposition intervention (`scripts/interventions.py`)
- BERT fine-tuning and checkpoint management (`models/bert_model.py`, `results/bert.json`)

**Ritvik Ganta**
- LLM zero-shot evaluation using BART-large-mnli (`models/llm_eval.py`, `results/llm_local.json`)
- Results overview notebook (`notebooks/results_overview.ipynb`)
- Final report writing and results summary

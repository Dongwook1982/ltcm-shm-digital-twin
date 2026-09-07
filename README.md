# LTCM Reproducibility Package

**Signal-Driven Lifecycle Memory for SHM Digital Twins: A Memory-Augmented Retrieval-Grounded Generation Framework with Physics-Informed Prediction and Multi-Infrastructure Validation**

Dong-Wook Kim · DL E&C Co., Ltd.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22591646.svg)](https://doi.org/10.5281/zenodo.22591646)
, Civil Smart Engineering Team, Seoul, Republic of Korea  
ORCID: [0000-0002-0721-5114](https://orcid.org/0000-0002-0721-5114) · clearup7@nate.com

*Manuscript submitted to Mechanical Systems and Signal Processing*  
Zenodo DOI (reproducibility package): [10.5281/zenodo.22591646](https://doi.org/10.5281/zenodo.22591646)

Manuscript DOI: *(to be assigned upon publication)*

---

## Contents

```
ltcm_repro/
├── README.md                          ← This file
├── config/
│   └── ltcm_config.yaml               ← All hyperparameters (Sec 4, Appendix C)
├── ontology/
│   ├── ltcm_infrastructure_ontology.ttl   ← OWL/Turtle ontology (Sec 3.3)
│   └── ltcm_ifc_rdf_mapping.csv       ← IFC 4.3 → RDF mapping table (38 types)
├── prompts/
│   └── marg_system_prompt.txt         ← GPT-4o system prompt template (Appendix B)
├── scripts/
│   ├── sparql_query_examples.rq       ← SPARQL 1.1 query examples (Sec 4.2)
│   └── pinn_training.py               ← PINN training script (Sec 5.1)
└── data/
    └── episodic_memory_schema_and_examples.json  ← Schema + 6 representative records
```

The supplementary benchmark queries are provided as a separate file:  
`LTCM_MSSP_S1_BenchmarkQueries.docx` (Supplementary Material S1, 40 queries)

---

## Quick Start

### 1. Environment

```bash
# Python 3.10+
pip install torch>=2.0 numpy>=1.24 scikit-learn>=1.3 scipy>=1.11 pyyaml rdflib

# For ontology validation (optional)
pip install owlready2
```

### 2. Run PINN Training (Synthetic Data)

Demonstrates the PINN training workflow and physics-informed loss formulation
using synthetic data with the same model structure and hyperparameter configuration
as the manuscript. **Synthetic outputs are illustrative and are not intended to
reproduce the numerical RMSE values obtained from the confidential operational
datasets** (see Data Availability statement in the manuscript).

```bash
# Track settlement — Sato-Odaka model (Case A)
python scripts/pinn_training.py --mechanism track_settlement --corpus_years 12

# Bearing degradation — Power-law model (Case C)
python scripts/pinn_training.py --mechanism bearing --corpus_years 32

# Corpus size sensitivity (corresponds to the corpus-size sensitivity workflow of Fig. 5 right panel)
for yr in 1 2 3 5 7 10 12; do
  python scripts/pinn_training.py --mechanism track_settlement --corpus_years $yr
done
```

**Note:** Synthetic outputs are illustrative. Values will differ from manuscript (operational data only).

### 3. Load Ontology

```python
from rdflib import Graph
g = Graph()
g.parse("ontology/ltcm_infrastructure_ontology.ttl", format="turtle")
print(f"Loaded {len(g)} triples")
# Expected: ~240 (schema TBox + key ABox examples)
```

### 4. Run Example SPARQL Queries

```bash
# Start Fuseki (Apache Jena Fuseki 4.9.0)
./fuseki-server --mem /ltcm

# Load ontology
curl -X POST http://localhost:3030/ltcm/data \
  --data-binary @ontology/ltcm_infrastructure_ontology.ttl \
  -H "Content-Type: text/turtle"

# Run one query at a time (copy a single SELECT block from sparql_query_examples.rq)
# Example with Query 1:
curl -X POST http://localhost:3030/ltcm/sparql \
  -H "Accept: application/json" \
  --data-urlencode "query=PREFIX ltcm: <https://ltcm.dlenc.kr/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?record_id ?timestamp WHERE {
  ?r a ltcm:EpisodicRecord ; rdfs:label ?record_id ; ltcm:hasTimestamp ?timestamp .
} LIMIT 10"
```

---

## Key Results (Reported in Manuscript — from Operational Data)

> **Note:** The values below are from the manuscript (trained on confidential
> operational data). The synthetic training script in this package demonstrates
> the same methodology but will produce different numerical outputs.

| Metric | Value |
|---|---|
| MAP@5 (full LTCM) | 0.913 |
| MAP@5 (baseline DT, stateless) | 0.641 |
| MAP@5 improvement | +42.4% |
| Multi-hop MARG vs. SPARQL-only | +40.3% (p < 0.001) |
| PINN RMSE — 12yr corpus (operational) | 1.4 yr |
| PINN RMSE — 3yr corpus (operational) | 3.1 yr |
| PINN RMSE improvement (corpus growth) | 55% |
| PINN RMSE improvement (physics regularization only) | 50% |
| Planning time reduction (N=32 engineers) | 36.4% (Cohen's d = 1.74) |
| Missed precedents reduction | 91% |

---

## System Configuration

See `config/ltcm_config.yaml` for complete hyperparameters. Key settings:

| Component | Setting |
|---|---|
| CLS-LTCM weights | w₁=0.45, w₂=0.35, w₃=0.20 |
| Significance threshold τ | 0.30 |
| MARG weights | α=0.25, β=0.35, γ=0.25, δ=0.15 |
| TransE embedding dim | 128 |
| KG size | 321,700 triples |
| Episodic records | 6,534 (post-consolidation) |
| LLM | gpt-4o-2024-08-06, zero-shot |
| Triplestore | Apache Jena Fuseki 4.9.0 (RDF-star) |
| Embedding framework | PyKEEN 1.10 |
| PINN optimizer | Adam (η=10⁻³) |
| Physics loss weight | λ_phys = 0.1 |
| PINN epochs / patience | 5,000 / 200 |
| MC Dropout passes | T = 100 |

---

## Data Availability

Operational monitoring data from the three infrastructure case studies (railway viaduct, expressway, PSC bridge) are confidential and cannot be made publicly available due to the data agreements with the respective infrastructure operators.

**What is provided:**
- Complete methodology (ontology, config, training scripts, prompts)
- Representative data examples illustrating the format (`episodic_memory_schema_and_examples.json`)
- Synthetic data generators in `pinn_training.py` that replicate corpus statistics
- 40 representative benchmark queries (Supplementary S1)

**What requires data access agreement:**
- Raw IoT sensor logs (12-year railway, 8-year road, 32-year bridge)
- Work order and inspection databases
- Full Knowledge Graph (321,700 triples, 6,534 episodic records)

Researchers wishing to validate on the operational data should contact the corresponding author at clearup7@nate.com.

---

## Citation

This manuscript is currently under review. A citable DOI will be added upon publication.
Please cite as:

> Kim, D.-W. (under review). Signal-Driven Lifecycle Memory for Structural Health
> Monitoring Digital Twins: A Memory-Augmented Retrieval-Grounded Generation Framework
> with Physics-Informed Prediction and Multi-Infrastructure Validation.
> *Submitted to Mechanical Systems and Signal Processing*.

The reproducibility package is available at:
https://github.com/Dongwook1982/ltcm-shm-digital-twin

---

## License

Code and ontology: **CC BY 4.0** (Creative Commons Attribution 4.0 International)  
Data examples: **CC BY 4.0** with the following note: operational records are representative examples only; actual operational data is confidential.

---

## Acknowledgements

The author thanks the three infrastructure operators who provided access to their maintenance records under confidentiality agreements, and the 32 maintenance engineers who participated in the controlled evaluation.

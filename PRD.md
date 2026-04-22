# Product Requirements Document
## Health Misinformation Detection System (MAVEN AI)
**Project:** UNC Department of Maternal and Child Health — Spring 2026
**Last Updated:** 2026-04-14
**Notebook reference:** `MAVEN_AI_UNC_SPR2026.ipynb`

---

## 1. Purpose

Build an NLP-driven classifier that automatically detects health misinformation in a targeted maternal health corpus and exposes the model through a lightweight dashboard or API endpoint for real-time text flagging.

---

## 2. Objectives

| # | Objective | Success Condition |
|---|---|---|
| O1 | Reliable NLP pipeline | Fine-tuned transformer model with documented, reproducible evaluation metrics |
| O2 | Domain-specific dataset | Labeled maternal health corpus assembled from scraped + existing sources |
| O3 | Functional interface | Dashboard or API endpoint processes arbitrary input text and returns classification |
| O4 | Rigor & accountability | Confusion matrix reviews at each milestone; out-of-sample stress test results documented; thresholds explicitly calibrated |

---

## 3. Scope

**In scope:**
- Binary or multi-class misinformation classification on maternal health text
- Data ingestion, annotation, and preprocessing pipeline
- Baseline classifier → advanced fine-tuned transformer
- Evaluation framework (metrics, error analysis, threshold tuning)
- MVP inference interface (dashboard or REST API)
- Final scaling proposal and handoff documentation

**Out of scope:**
- Real-time social media monitoring or streaming ingestion
- Multilingual support
- Medical advice or clinical decision support functionality

---

## 4. Stakeholders

| Role | Name / Group |
|---|---|
| Faculty Advisor | Dr. Bazzano |
| Research Team | AI@UNC project team |
| End Users (demo) | Dept. of MCH researchers / reviewers |

---

## 5. Functional Requirements

### 5.1 Data & Preprocessing

| ID | Requirement | Priority |
|---|---|---|
| D1 | Ingest at least one established misinformation benchmark dataset | Must |
| D2 | Scrape and store a targeted maternal health corpus from identified sources | Must |
| D3 | Define and document annotation guidelines collaboratively with Dr. Bazzano | Must |
| D4 | Label a minimum viable annotated subset before Milestone 1 | Must |
| D5 | Implement a text preprocessing pipeline using established techniques (tokenization, normalization, stopword handling) | Must |
| D6 | Continuously expand and re-label corpus through Milestone 2 | Should |
| D7 | Version and persist all dataset snapshots for reproducibility | Should |

**Constraint:** Do not over-iterate on custom preprocessing. Prefer well-validated, published techniques to reduce risk.

### 5.2 Model Development

| ID | Requirement | Priority |
|---|---|---|
| M1 | Train a baseline classifier (logistic regression or lightly fine-tuned transformer) | Must |
| M2 | Establish a Hugging Face transformer pipeline as the primary model architecture | Must |
| M3 | Document all hyperparameter choices and training configurations | Must |
| M4 | Evaluate at least two distinct model architectures or fine-tuning strategies | Should |
| M5 | Select final architecture based on evaluation metrics agreed upon with Dr. Bazzano | Must |

### 5.3 Evaluation & Quality Assurance

| ID | Requirement | Priority |
|---|---|---|
| E1 | Define core evaluation metrics (e.g., F1, precision, recall, AUC) before Milestone 1 in consultation with Dr. Bazzano | Must |
| E2 | Produce confusion matrices at every milestone checkpoint | Must |
| E3 | Conduct out-of-sample robustness testing on held-out edge cases (Milestone 4) | Must |
| E4 | Tune classification threshold to balance false positive and false negative rates; document chosen threshold and rationale | Must |
| E5 | Reference AI@UNC prior project methodologies for error analysis guidance | Should |

### 5.4 Deployment & Interface

| ID | Requirement | Priority |
|---|---|---|
| I1 | Build an MVP dashboard **or** REST API endpoint that accepts raw text and returns a misinformation flag + confidence score | Must |
| I2 | Interface must process new input in real time (< 5 s response for a single text input) | Must |
| I3 | Interface must be demonstrable without local GPU (CPU inference acceptable for demo) | Must |
| I4 | Provide a brief user guide for operating the interface | Should |

### 5.5 Documentation & Handoff

| ID | Requirement | Priority |
|---|---|---|
| Doc1 | Maintain running documentation covering dataset decisions, model architecture, and evaluation results | Must |
| Doc2 | Produce a stress-testing methodology report by Milestone 4 | Must |
| Doc3 | Deliver a formalized scaling proposal addressing future dataset validation and model productionization | Must |

---

## 6. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Reproducibility | All experiments must be runnable from a clean Colab environment using pinned dependency versions |
| Transparency | Model predictions must include a confidence score; threshold rationale must be documented |
| Safety | Model outputs must not be presented as medical advice in any interface copy |

---

## 7. Implementation Timeline

### Milestone 1 — Weeks 1–2: Foundation
- [ ] Source and ingest at least one existing misinformation dataset
- [ ] Finalize annotation guidelines with Dr. Bazzano
- [ ] Build baseline text preprocessing pipeline
- [ ] Train and evaluate baseline classifier
- [ ] Agree on core evaluation metrics

**Gate:** Baseline model with documented metrics exists.

---

### Milestone 2 — Weeks 3–5: Advanced Model
- [ ] Expand and re-label the maternal health corpus
- [ ] Implement Hugging Face fine-tuned transformer
- [ ] Complete error analysis (confusion matrix review)
- [ ] Apply and document advanced fine-tuning strategies
- [ ] Compare at least two architectures

**Gate:** Transformer model outperforms baseline on agreed metrics.

---

### Milestone 3 — Week 6: MVP Interface
- [ ] Build dashboard or API endpoint for real-time text flagging
- [ ] Validate end-to-end: input text → classification output
- [ ] Deploy to shareable environment (Colab, HuggingFace Spaces, or hosted API)

**Gate:** A non-technical reviewer can flag text using the interface without local setup.

---

### Milestone 4 — Weeks 7–8: Robustness & Calibration
- [ ] Execute out-of-sample stress tests targeting edge cases
- [ ] Document stress-testing methodology and results
- [ ] Tune and lock final classification threshold
- [ ] Produce calibrated precision/recall tradeoff analysis

**Gate:** Out-of-sample performance meets or exceeds thresholds agreed with Dr. Bazzano; threshold choice is documented.

---

### Milestone 5 — Week 9: Final Delivery
- [ ] Finalize and freeze model
- [ ] Deliver demo interface
- [ ] Complete all documentation
- [ ] Submit scaling proposal

**Gate:** All deliverables handed off; project can be resumed by a new team without author involvement.

---

## 8. Open Questions

| # | Question | Owner |
|---|---|---|
| Q1 | What are the precise metric thresholds that define acceptable model performance? | Dr. Bazzano + team |
| Q2 | Which sources will comprise the maternal health corpus? (Candidates include Wikipedia, Harvard Health, PubMed, WHO, CDC, and others.) | Research team |
| Q3 | Dashboard (frontend) vs. REST API — which interface form factor is preferred for the final demo? | Dr. Bazzano |
| Q4 | Are there IRB or data-use constraints on any scraped or labeled data? | Dr. Bazzano |
| Q5 | What counts as the "maternal health" label boundary — postpartum, prenatal, both, broader women's health? | Dr. Bazzano + team |

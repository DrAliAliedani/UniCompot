# UniCompot: An Arabic Retrieval-Augmented Assistant for University Regulations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrAliAliedani/UniCompot/blob/main/notebooks/UniCompot_RAG_and_Evaluation.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

This repository contains the official implementation and evaluation code for the paper:

> **<PAPER TITLE>**
> <Author 1>, <Author 2>, ... — University of Basrah, Iraq
> *<Journal / Conference>, <Year>*. DOI: <DOI>

## Introduction

Students and staff at Iraqi universities must navigate a large body of official Arabic regulations — examination and disciplinary rules, student affairs, academic freedom and employee rights, gender equality, and environmental policies. These documents are long, scattered across PDFs, and written in formal legal Arabic, which makes finding a precise answer slow and error-prone.

**UniCompot** is a retrieval-augmented generation (RAG) system (Lewis et al., 2020) that answers student questions in Arabic, grounded in the official regulations of the University of Basrah. It combines:

- **Hybrid retrieval** — dense semantic search with multilingual E5 embeddings (Wang et al., 2024) over a FAISS HNSW index (Johnson et al., 2019; Malkov & Yashunin, 2020), fused with sparse BM25 keyword search (Robertson & Zaragoza, 2009) through a tunable weight α.
- **Cross-encoder reranking** with BGE-reranker-v2-m3 (Chen et al., 2024).
- **Metadata routing** — a single canonical Arabic keyword table tags every chunk at indexing time and routes every query at question time, so queries are only routed to categories that actually exist in the index.
- **Arabic-centric generation** with Jais-2-8B-Chat (Anwar et al., 2025) loaded in 4-bit NF4 quantisation (Dettmers et al., 2023) to run on a single GPU.

The repository also contains the full **evaluation protocol** used in the paper:

| Aspect | Metric(s) | Reference |
|---|---|---|
| Retrieval | Recall@5, Recall@10, MRR@10, nDCG@10 | Järvelin & Kekäläinen (2002); Voorhees (1999) |
| Answer faithfulness | Sentence-level NLI entailment + hallucination rate (mDeBERTa-v3 XNLI) | He et al. (2023); Conneau et al. (2018); Laurer et al. (2024) |
| Answer relevance | Question–answer cosine similarity (E5) | Wang et al. (2024) |
| Citation grounding | Fraction of cited article numbers ("المادة N") present in the retrieved context | — (this work) |
| Routing | Accuracy + confusion matrix | — |
| Baseline | Closed-book (no-RAG) generation | Roberts et al. (2020) |
| Significance | Paired bootstrap test (10,000 resamples) | Koehn (2004); Smucker et al. (2007) |
| Human evaluation | Blind rating sheet (correctness, groundedness, completeness; 1–5) + Cohen's κ | Howcroft et al. (2020); Cohen (1960); Es et al. (2024) |

## System overview

```
                ┌──────────────┐
 PDF corpus ──► │ Chunking      │ 2000 chars, 300 overlap
                │ + category    │ canonical CATEGORY_KEYWORDS
                └──────┬───────┘
          ┌────────────┴────────────┐
   FAISS HNSW (E5-large)       BM25 (rank-bm25)
          └────────────┬────────────┘
         Hybrid fusion: α·semantic + (1−α)·BM25   ◄── query routing (route_query)
                       │
         BGE-reranker-v2-m3 (top-k = 6)
                       │
         Jais-2-8B-Chat (4-bit)  ──►  Answer
                       │
   Evaluation: NLI faithfulness · relevance · citation accuracy · IR metrics
```

## Repository structure

```
UniCompot/
├── notebooks/
│   └── UniCompot_RAG_and_Evaluation.ipynb   # full pipeline + evaluation (run top to bottom)
├── data/
│   └── README.md                            # how to obtain / place the regulations corpus
├── results/                                 # CSV / XLSX / JSON outputs are written here
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Requirements

- **GPU**: ≥16 GB VRAM (T4 works but is tight; L4 or A100 recommended). The 4-bit 8B LLM, the E5-large embedder, the BGE reranker and the mDeBERTa NLI model are all resident simultaneously.
- **Hugging Face account** with access to [`inceptionai/Jais-2-8B-Chat`](https://huggingface.co/inceptionai/Jais-2-8B-Chat) (accept the licence on the model page) and an access token.
- **Python** ≥ 3.10.

## Quick start

### Option A — Google Colab (recommended)

1. Click the **Open in Colab** badge above.
2. `Runtime → Change runtime type → GPU`.
3. Add your Hugging Face token as a Colab secret named `HF_TOKEN` (key icon in the left sidebar) and grant the notebook access.
4. Upload the regulations corpus as `/content/allfile.pdf` (see [`data/README.md`](data/README.md)).
5. Run all cells top to bottom. The Gradio demo cell (Section 16) is optional and blocks the notebook until stopped.

### Option B — Local machine

```bash
git clone https://github.com/DrAliAliedani/UniCompot.git
cd UniCompot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login            # paste your HF token
jupyter notebook notebooks/UniCompot_RAG_and_Evaluation.ipynb
```

When running locally, replace the Colab-specific login cell (`google.colab.userdata`) with `huggingface_hub.login()` and change the PDF path `/content/allfile.pdf` to `data/allfile.pdf`.

## Reproducing the paper results

All reported numbers are computed on one **fixed, stratified evaluation set** (`n = 60`, `seed = 42`), saved as `fixed_eval_set_n60_seed42.json` so every configuration is scored on identical queries.

| Notebook section | Output file | Paper content |
|---|---|---|
| 24 – Retrieval comparison (semantic / BM25 / hybrid α ∈ {0.3, 0.5, 0.7} / ± reranker) | `unified_retrieval_comparison.csv` | Retrieval table |
| 25 – Full RAG evaluation (best configuration) | `rag_full_eval_results.csv` | Faithfulness, relevance, hallucination, citation accuracy |
| 26 – Closed-book baseline | `closed_book_baseline_results.csv` | No-RAG baseline |
| 27 – Routing accuracy | printed confusion matrix | Routing analysis |
| 28 – Significance test | printed mean diff, 95% CI, *p* | Statistical significance |
| 29 – Human evaluation sheet | `human_eval_sheet_BLIND.xlsx`, `human_eval_answer_key.csv` | Human evaluation, Cohen's κ |

**Note on determinism.** Generation uses sampling (`temperature = 0.3`, `top_p = 0.9`) and evaluation questions are generated by the LLM, so absolute numbers may vary slightly between runs and GPU types. Retrieval metrics on a saved evaluation set are deterministic. To reproduce our exact queries, load the released `fixed_eval_set_n60_seed42.json` instead of regenerating it.

## Models used

| Role | Model | Licence |
|---|---|---|
| Generator | [`inceptionai/Jais-2-8B-Chat`](https://huggingface.co/inceptionai/Jais-2-8B-Chat) | Apache-2.0 |
| Embeddings | [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) | MIT |
| Reranker | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Apache-2.0 |
| NLI (faithfulness) | [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | MIT |

Model weights are downloaded from the Hugging Face Hub at run time and are **not** redistributed in this repository; each remains under its own licence.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{unicompot2026,
  title   = {<PAPER TITLE>},
  author  = {<Surname1>, <Name1> and <Surname2>, <Name2>},
  journal = {<Journal>},
  year    = {2026},
  doi     = {<DOI>},
  note    = {Code: https://github.com/DrAliAliedani/UniCompot}
}
```

## References

- Anwar, M., Freihat, A., Ibrahim, G., et al. (2025). *Jais 2: A Family of Arabic-Centric Open Large Language Models*. Technical Report, MBZUAI / Inception / Cerebras. arXiv:2608.13580.
- Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *Findings of ACL 2024*, 2318–2335. arXiv:2402.03216.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., & Stoyanov, V. (2018). XNLI: Evaluating cross-lingual sentence representations. *Proceedings of EMNLP 2018*, 2475–2485.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of EACL 2024: System Demonstrations*, 150–158.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*.
- Howcroft, D. M., Belz, A., Clinciu, M., et al. (2020). Twenty years of confusion in human evaluation: NLG needs evaluation sheets and standardised definitions. *Proceedings of INLG 2020*, 169–182.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422–446.
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
- Koehn, P. (2004). Statistical significance tests for machine translation evaluation. *Proceedings of EMNLP 2004*, 388–395.
- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. *Political Analysis*, 32(1), 84–100.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 9459–9474.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836.
- Roberts, A., Raffel, C., & Shazeer, N. (2020). How much knowledge can you pack into the parameters of a language model? *Proceedings of EMNLP 2020*, 5418–5426.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Smucker, M. D., Allan, J., & Carterette, B. (2007). A comparison of statistical significance tests for information retrieval evaluation. *Proceedings of CIKM 2007*, 623–632.
- Voorhees, E. M. (1999). The TREC-8 question answering track report. *Proceedings of TREC-8*, 77–82.
- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. arXiv:2402.05672.

## License

The code in this repository is released under the [MIT License](LICENSE). The regulation documents belong to the University of Basrah / the Iraqi Ministry of Higher Education and Scientific Research and are not covered by this licence.

## Contact

For questions, please open a GitHub issue or contact <Corresponding author> (ali.nabeel@uobasrah.edu.iq), University of Basrah, Iraq.

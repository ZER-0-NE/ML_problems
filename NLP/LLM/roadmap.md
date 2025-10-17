Perfect — you’re essentially asking for a **chronological roadmap** that organizes the *evolution of methods* (from classic to cutting-edge, e.g. DeepSeek, GRPO) **within each stage of the LLM training lifecycle**, applied to your tiny experiment framework.

We’ll treat each phase — *tokenization → pretraining → midtraining → SFT → RLHF → inference (and beyond)* — as a container of methods, **sorted oldest → newest**, each building on the previous.

The goal: **understand the lineage of architectural & algorithmic advances** and how you’d extend your small “spicy–chili” transformer through each wave.

---


# 🔭 Unified LLM Evolution Roadmap (Architecture + Training)

| Phase                                            | Chronological Method / Breakthrough                               | Type         | Summary / Goal                                       | Example (Tiny “spicy–chili” experiment)                     |
| ------------------------------------------------ | ----------------------------------------------------------------- | ------------ | ---------------------------------------------------- | ----------------------------------------------------------- |
| **1️⃣ Tokenization & Data Handling**             | Word-level tokenization (GPT-1, 2018)                             | Algorithm    | Basic text → indices                                 | Start baseline with word-level tokens                       |
|                                                  | Subword/BPE (GPT-2/3, 2019–2020)                                  | Algorithm    | Frequency-based subword segmentation                 | Tokenize “spicy ↔ chili” with BPE                           |
|                                                  | WordPiece (BERT, 2018–2019)                                       | Algorithm    | Likelihood-based subword vocabulary                  | Compare embedding similarity to BPE                         |
|                                                  | Byte-Level BPE (GPT-2)                                            | Algorithm    | Operates on bytes → universal tokenization           | Ensure all chars in toy corpus are tokenized                |
|                                                  | SentencePiece / Unigram LM (T5, 2020)                             | Algorithm    | Subword tokenization via probabilistic unigram model | Optional: see if dynamic vocab helps tiny corpus            |
|                                                  | Adaptive / dynamic tokenization (2024+)                           | Algorithm    | Adjust vocab during training                         | Try learning a mini-vocab optimized for “spicy–chili”       |
| **2️⃣ Pre-training / Foundational Stage**        | Classic decoder-only next-token prediction (GPT-1, 2018)          | Algorithm    | Learn general token stats                            | Baseline autoregressive LM on 19 sentences                  |
|                                                  | Masked LM / denoising (BERT, T5, 2019)                            | Algorithm    | Learn bidirectional context                          | Mask “chili” and predict                                    |
|                                                  | Encoder–Decoder (T5/BART, 2019–2020)                              | Architecture | Full seq2seq pipeline                                | Compare causal-only vs seq2seq output                       |
|                                                  | Mixture of denoising + causal (UniLM, ELECTRA, 2020)              | Algorithm    | Multi-objective LM                                   | Combine masked + autoregressive loss                        |
|                                                  | Sparse / Long-context attention (Longformer, DeepSeek, 2025)      | Architecture | Efficient reasoning over long sequences              | Test block-sparse attention on extended toy sentences       |
|                                                  | Curriculum pretraining & data mixture (Chinchilla-style, 2022)    | Algorithm    | Focus learning on high-value patterns                | Mix synthetic paraphrases with original sentences           |
|                                                  | Self-evolution / synthetic data bootstrapping (DeepSeek-R1, 2025) | Algorithm    | Model generates own data for pretraining             | Have model paraphrase “spicy ↔ chili” pairs                 |
| **3️⃣ Mid-training / Representation Refinement** | Adapter-tuning (Houlsby et al., 2019)                             | Algorithm    | Lightweight refinement                               | Add adapter layers for “spicy–chili” mapping                |
|                                                  | LoRA / QLoRA (2022–2023)                                          | Algorithm    | Low-rank fine-tuning                                 | Apply LoRA to 2-layer transformer                           |
|                                                  | Prefix-tuning / P-tuning v2 (2022)                                | Algorithm    | Learn task-specific prefixes                         | Optional: prepended tokens for spicy task                   |
|                                                  | Contrastive embedding alignment (SimCSE, 2023)                    | Algorithm    | Bring related concepts closer                        | Force “spicy” ↔ “chili” embeddings together                 |
|                                                  | Mixture-of-Experts routing (2023–2024)                            | Algorithm    | Conditional computation                              | Experiment with tiny 2-expert MoE                           |
|                                                  | Sparse / structured routing (DeepSeek, 2025)                      | Algorithm    | Efficient reasoning                                  | Route attention to relevant words only                      |
| **4️⃣ SFT (Supervised Fine-tuning)**             | Classic instruction SFT (InstructGPT, 2022)                       | Algorithm    | Teach task behavior                                  | “What food is spicy?” → “chili”                             |
|                                                  | Data-mix ratio tuning (Alpaca, Dolly, 2023)                       | Algorithm    | Balance multiple datasets                            | Mix paraphrases and original Q&A                            |
|                                                  | Domain-adaptive SFT (custom tasks, 2024)                          | Algorithm    | Focused domain adaptation                            | Only culinary flavor Q&A                                    |
|                                                  | Chain-of-Thought SFT (2024–2025)                                  | Algorithm    | Teach stepwise reasoning                             | “Spicy → hot → chili” reasoning                             |
|                                                  | Self-distilled / teacher-student SFT (2025+)                      | Algorithm    | Compress knowledge                                   | Distill 32-dim student model                                |
| **5️⃣ Alignment / RLHF / RLAIF**                 | PPO-based RLHF (2022)                                             | Algorithm    | Align to human feedback                              | Reward output containing “chili”                            |
|                                                  | DPO (Direct Preference Optimization, 2023)                        | Algorithm    | Preference learning                                  | Adjust small-batch gradients                                |
|                                                  | ORPO / RRHF (2023–2024)                                           | Algorithm    | Ranking-based RL                                     | Rank multiple candidate outputs                             |
|                                                  | GRPO (2025)                                                       | Algorithm    | Grouped reward optimization                          | Stabilize mini-batch learning                               |
|                                                  | Self-rewarding / automatic feedback (DeepSeek-R1, 2025)           | Algorithm    | Model evaluates itself                               | Generate paraphrase, reward if correct                      |
| **6️⃣ Reasoning / RL-based Enhancement**         | Step-by-step reasoning prompts (CoT, 2022)                        | Algorithm    | Multi-step logic                                     | Explicit intermediate reasoning in outputs                  |
|                                                  | Self-consistency sampling (2023)                                  | Algorithm    | Reduce reasoning errors                              | Sample multiple CoT outputs for “spicy → chili”             |
|                                                  | Toolformer / API-calling (2023)                                   | Algorithm    | External tool integration                            | Optional small toy API call                                 |
|                                                  | Reasoning RL (DeepSeek, 2025)                                     | Algorithm    | Reward logical inference                             | Reward correct CoT chains                                   |
|                                                  | Multi-objective RL (2025)                                         | Algorithm    | Safety + reasoning                                   | Penalize nonsensical outputs                                |
| **7️⃣ Retrieval / External Memory**              | RAG (2020)                                                        | Algorithm    | Retrieve documents at query time                     | Retrieve original 19 sentences for context                  |
|                                                  | kNN-LM (2021)                                                     | Algorithm    | Nearest-neighbor LM                                  | Embed toy corpus, query nearest                             |
|                                                  | Self-Retriever (2023)                                             | Algorithm    | Model-driven retrieval                               | Auto-select relevant “spicy” examples                       |
|                                                  | Memory Transformer / KV-cache (2024)                              | Architecture | Persistent memory for reasoning                      | Store token K/V for incremental inference                   |
|                                                  | Adaptive retrieval controllers (2025)                             | Algorithm    | Dynamic memory access                                | Feed top-k retrievals to query                              |
| **8️⃣ Post-training Optimization / Compression** | Distillation (2015+)                                              | Algorithm    | Reduce model size                                    | Student model predicts “chili” correctly                    |
|                                                  | Quantization (INT8, 4-bit, 2020–2023)                             | Algorithm    | Reduce memory/computation                            | Tiny int8 transformer                                       |
|                                                  | Mixture-of-Experts pruning (2023)                                 | Algorithm    | Remove unused experts                                | Optional MoE pruning                                        |
|                                                  | Low-rank pruning / activation sparsity (2024)                     | Algorithm    | Reduce compute                                       | Apply LoRA + pruning                                        |
|                                                  | Dynamic quantization & mixed-precision (2025)                     | Algorithm    | Optimized inference                                  | 16-bit / 8-bit mixture                                      |
| **9️⃣ Inference & Deployment**                   | Greedy / beam search                                              | Algorithm    | Baseline decoding                                    | Generate “spicy → chili” outputs                            |
|                                                  | Top-k / nucleus sampling (GPT-2, 2019)                            | Algorithm    | Stochastic decoding                                  | Compare sample quality                                      |
|                                                  | Temperature & repetition penalty                                  | Algorithm    | Control diversity                                    | Tune for toy corpus                                         |
|                                                  | Contrastive decoding (2023)                                       | Algorithm    | Avoid hallucination                                  | Reward embedding alignment                                  |
|                                                  | Reinforced / planning-based decoding (2024–2025)                  | Algorithm    | Align output with RL                                 | Use GRPO reward during decoding                             |
|                                                  | Speculative decoding (2023)                                       | Architecture | Two-model draft → verify                             | Small “draft” model predicts tokens, verified by main model |
|                                                  | KV Cache reuse / incremental inference                            | Architecture | Fast token-by-token generation                       | Implement for tiny transformer                              |
| **🔟 Evaluation / Interpretability**             | Perplexity                                                        | Algorithm    | Classic LM metric                                    | Track baseline loss                                         |
|                                                  | Embedding similarity probing                                      | Algorithm    | Test learned associations                            | Cosine between “spicy” ↔ “chili”                            |
|                                                  | Attention visualization                                           | Algorithm    | Understand learned dependencies                      | Plot attention heads linking flavor words                   |
|                                                  | Behavioral metrics (helpfulness, safety)                          | Algorithm    | Align model behavior                                 | Reward correct Q&A outputs                                  |
|                                                  | Latent concept tracing (2025)                                     | Algorithm    | Track abstract learned concepts                      | Trace “spicy” vector evolution during training              |

---


| Era           | Breakthrough                                        | Core Benefit                             | Paradigm            |
| ------------- | --------------------------------------------------- | ---------------------------------------- | ------------------- |
| **2017–2018** | Transformer (Vaswani), Absolute Positional Encoding | Core architecture                        | Foundation          |
| **2018–2019** | WordPiece (BERT), Byte-Level BPE (GPT-2)            | Robust tokenization                      | Representation      |
| **2019–2020** | Encoder–Decoder (T5/BART)                           | Bidirectional understanding + generation | Structural          |
| **2021**      | Rotary Positional Embedding (RoPE)                  | Long-context scaling                     | Attention geometry  |
| **2022**      | FlashAttention + KV Cache                           | Efficient inference & training           | System architecture |
| **2022**      | ALiBi                                               | Positional generalization                | Attention biasing   |
| **2023**      | Speculative Decoding                                | Latency breakthrough                     | Inference           |
| **2023–2025** | Sparse / Structured Attention (DeepSeek-R1)         | Long-context + reasoning efficiency      | Adaptive routing    |


---

### ✅ Notes on This Table

1. **Architecture vs Algorithm**: All hardware/structural innovations (RoPE, ALiBi, FlashAttention, KV cache, encoder–decoder, speculative decoding) are explicitly labeled **Architecture**. Training/objective improvements remain **Algorithm**.
2. **Tiny experiment mapping**: For each row, I included what you can try on your mini “spicy–chili” corpus. This ensures even a 19-sentence toy dataset can illustrate major trends.
3. **Chronological flow preserved**: Everything is listed in *approximate adoption order*, so you can see lineage from GPT-1 → DeepSeek-R1 (2025).

---



## ⚙️ How to Use This Roadmap

1. **Treat each phase as a module** you can plug into your existing small transformer.
2. **Incrementally update** one dimension at a time — tokenization, objective, or attention type — while keeping dataset and evaluation constant.
3. **Track metrics**: training/validation perplexity, success rate on “spicy–chili”, embedding cosine similarity, and attention map interpretability.
4. **Document the lineage**: note how each innovation (LoRA → sparse attention → GRPO) alters learning behavior even on tiny data.

---


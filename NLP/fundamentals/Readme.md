# Quick Notes

- Sentences can first be split into words (or subword units) called **tokens** using **tokenization**
- These tokens are then assigned an integer value called a **token ID**, which can be converted into a one-hot encoded vector as shown later.
- The process of assigning any non-numeric data, such as images and text, a numerical representation is called **embedding**, and so these vector representations of words are known as **word embeddings**. 
- **Word Embedding **is the process of taking a word and creating a vector representation in N-dimensional space. 
- Prior to 2013, word embeddings were often created using one-hot encoding. This method for producing vector representations is very simple: for each word construct a vector with 0s in every element, except at the position equal to the token ID which should be filled with a 1. This creates a unique vector for each word, where the position of the 1 indicates which word is being encoded (hence the name ‘one-hot’). Because of this, one-hot vectors are called localist representations, as all the information that represents the word is restricted to a single element.
- The collection of words a model can encode is called the **vocabulary**, and number of words in the vocabulary is called the **vocabulary size**.
- **word2vec** is a family of algorithms that produce distributed word embeddings for use in NLP tasks. These vectors are far denser than those created using the one-hot encoding method (i.e. very few, if any, of the elements are 0), and so they can be much smaller in size. The idea is to create an N-dimensional vector space, in which similar words are geographically close to each other. 
- Typically, these embeddings have around 300 dimensions. Once these embeddings are created, they can be written to a file and loaded into memory when needed to essentially form a lookup table at run time. When a language model is given some input text, the text is first converted into tokens. 
- These are then converted into vectors by finding the appropriate row in the word2vec embeddings matrix. For this reason, the embeddings produced by word2vec are called static. These static embeddings form the basis for the so-called dynamic or contextual embeddings that are used in LLMs, which are made by adding context from the surrounding sentences or paragraphs to each word.

# Tokenizer

## Q Why train a tokenizer like toktoken on your own data? what kind of existing tokenizer do we have?

Ollama 3 models (including Llama 3.0, 3.1, 3.2, and 3.3) use a **Byte-Pair Encoding (BPE)** tokenizer built on top of **tiktoken**, the same base used by GPT‑2 and OpenAI models—not SentencePiece like Llama 2 ([huggingface.co][1]).

---

#### 🔍 Key Details

* **Based on tiktoken’s BPE**: Unlike Llama 2’s SentencePiece tokenizer, Llama 3 relies on a BPE format similar to GPT‑2’s, implemented using tiktoken .
* **Vocabulary size ≈ 128,256 tokens**: This large token set encompasses subwords and whole words, enabling efficient encoding.
* **Smart merging rule**: It respects vocabulary membership—if a complete word like `"hugging"` exists in the vocab, it's used as one token rather than splitting into smaller parts ([ollama.com][2], [huggingface.co][1]).
* **GGML metadata confirms format**: Ollama's GGML files specify `tokenizer.ggml.model = gpt2` and `tokenizer.ggml.pre = llama-bpe`, reaffirming the use of BPE and GPT‑2-style token handling ([ollama.com][3]).

---

#### 🧩 Example: "hugging"

* **If** `"hugging"` is in the vocab → ✅ Encoded as a single token.
* **Otherwise**, it's split via BPE: `"hug"` + `"ging"` = 2 tokens.

This policy helps reduce token count and improve encoding efficiency.

[1]: https://huggingface.co/docs/transformers/en/model_doc/llama3?utm_source=chatgpt.com "Llama3 - Hugging Face"
[2]: https://ollama.com/library/llama3.3/blobs/4824460d29f2?utm_source=chatgpt.com "llama3.3/model - Ollama"
[3]: https://ollama.com/library/llama3.2/blobs/dde5aa3fc5ff?utm_source=chatgpt.com "llama3.2/model - Ollama"


---

### Q what does o3 model use for its tokenizer?

Tokenizer & Context Capacity
- Uses OpenAI’s tiktoken-style BPE tokenizer (GPT‑2-based), suitable for large conditional reasoning; context window reliably spans ~25–64K tokens per user reports 

- While advertised with high context support, real-world usage tops out around 25K tokens before quality drops. 


### Q on which data are these tokenizer trained on?

OpenAI hasn’t publicly disclosed the **precise training corpus** for the o3 tokenizer. However, we can make well-informed inferences based on their common practices and recent research into tokenizer data composition:

---

#### 🔍 What We Know

##### • OpenAI tokens are typically trained on large-scale web and code datasets

OpenAI’s earlier models (GPT‑3, GPT‑3.5, GPT‑4) were trained on massive mixtures of web text (Common Crawl, curated news, books), code repositories (e.g., GitHub) and other curated sources. It’s very likely o3 follows this same diversified approach.

##### • BPE tokenizers reveal data composition

A recent academic study “Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?” found that by analyzing the learned merge rules of BPE tokenizers, you can infer the relative proportions of code vs. natural language in the training set ([openai.com][4], [arxiv.org][5]).

They observed that:

* GPT‑3.5 and Claude tokenizers were \~60 % code-centric.
* GPT‑4o and Mistral’s tokenizers reflected high multilingual text coverage.
  The study found Llama 3 tokenizer to be \~48 % multilingual ([arxiv.org][5]).

By extension, OpenAI’s o3 likely also trained on a mixed dataset with:

* **Significant web text** (English → multilingual).
* **A substantial amount of code**, based on their model’s strong coding performance noted in benchmarks ([techtarget.com][6]).

---

#### 🧠 Why It Matters

| Data Type    | Likely Proportion      | Influence on o3                             |
| ------------ | ---------------------- | ------------------------------------------- |
| Web/text     | Major portion          | Strong language & reasoning                 |
| Code         | Large slice (\~50–60%) | Enables reliable code generation & analysis |
| Multilingual | Moderate to high       | Reflects multilingual benchmarks and usage  |

---

* **Exact sources**: Not released, but likely include Common Crawl, curated books, news, GitHub code, and possibly specialized domains.
* **Inferred composition**: A balanced mix of web text (for reasoning and diverse language) plus heavy code representation (to power STEM/code benchmarks).
* Academic insights on BPE token splits help triangulate that o3’s tokenizer follows similar data patterns to previous OpenAI models ([en.wikipedia.org][7], [visla.us][8], [openai.com][1], [arxiv.org][5]).

---


[4]: https://openai.com/index/introducing-o3-and-o4-mini/?utm_source=chatgpt.com "Introducing OpenAI o3 and o4-mini"
[5]: https://arxiv.org/abs/2407.16607?utm_source=chatgpt.com "Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?"
[6]: https://www.techtarget.com/whatis/feature/OpenAI-o3-explained-Everything-you-need-to-know?utm_source=chatgpt.com "OpenAI o3 and o4 explained: Everything you need to know"
[7]: https://en.wikipedia.org/wiki/OpenAI_o3?utm_source=chatgpt.com "OpenAI o3"
[8]: https://www.visla.us/blog/news/openai-o3-and-o4-mini-openais-new-models-explained/?utm_source=chatgpt.com "OpenAI o3 and o4-mini: OpenAI's new models, explained - Visla"



## Q can we use the same tokenizer that was used for these models? are they open source? what is the best tokenizer we can have for web text

1. OpenAI’s tiktoken – the official tokenizer
tiktoken is OpenAI’s Rust-backed BPE tokenizer, used in GPT‑3, GPT‑4, and o‑series models.

- https://github.com/openai/tiktoken

It supports encodings like o200k_base (used by o‑series) and cl100k_base (GPT‑4/GPT‑3.5) 
news.ycombinator.com


It's open source, fast, reversible, production-ready, and available for Python (and unofficial ports exist for other languages).

Example in Python:

```
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
tokens = enc.encode("Hello, world!")
```


#### Q what does encode do here? will it create it into numerical representation based on o200k_base pre-trained tokenizer? if say hello = 15 and world = 23, then will this return an array of [15, 23], in very simple terms?

When you call:

```
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
tokens = enc.encode("Hello, world!")
```

here’s exactly what happens:

🔍 What .encode() Does
- Converts text into a list of token IDs – essentially turning strings into numerical representations.

- It uses the o200k_base BPE vocabulary and merge rules, the same scheme used by OpenAI’s o-series models - https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

- The output is an integer array where each element represents a subword or token recognized by the tokenizer.

🧠 Simple Example
Let’s say, hypothetically:

"Hello" maps to token ID 15

", " maps to ID 42

"world" to 23, and

"!" to 7

Then:

```
tokens = enc.encode("Hello, world!")
# => [15, 42, 23, 7]
```

Yes—it’s exactly like your simplified [15, 23] example, but with more precision and including punctuation/subword tokens.

✅ Why This Matters
Token IDs are what the model actually reads and processes.

encode(...) is deterministic and reversible, meaning enc.decode(tokens) returns the exact original text .

Useful for:

Computing token length (to manage model context limits)

Converting prompts into model-readable integer inputs

Performing preprocessing or batching



## Q what does this mean to train custom BPE on your data? will this be the training data or something else? I thought tokenizers will almost always cover all the words that exist in the dictionary?

Training a **custom BPE tokenizer** means **learning a vocabulary and merge rules directly from your own dataset**, not using a pre-existing tokenizer. It's a separate process from training the language model itself.

---

#### 🔧 What "Training a custom BPE tokenizer" Means

1. **Collect your text data** (e.g., all your emails, documents, tweets).
2. **Normalize and pre-tokenize**—clean text and split into words or byte-level tokens.
3. **Initialize** a BPE tokenizer (e.g., from Hugging Face’s `tokenizers` library).
4. **Train** it on your corpus to:

   * Identify frequent byte sequences.
   * Iteratively merge those pairs to build up your vocabulary to a target size (e.g., 30K–100K tokens).
     ([huggingface.co][9], [huggingface.co][10], [medium.com][11])
5. **Produce** two key outputs:

   * A list of tokens (vocabulary).
   * A list of merge rules (pairs of tokens that should fuse into one).

Once that's trained, you use it into your code:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]"])
tokenizer.train(["my_corpus.txt"], trainer)
```



---

#### 📚 Is It the Same as Your Model Training Data?

No—**tokenizer training only builds the vocabulary**, not the language model's weights. You can train a tokenizer on the same data as your model, or even on entirely different data suited to your domain.

Example:

* You have **legal documents** → train BPE on that corpus.
* Use this tokenizer with a **pretrained LLaMA/FLAN** model so it encodes legal text more efficiently.

---

#### ❓Why Do It?

* **Better token coverage** for your domain (e.g., medical jargon, code snippets).
* **Fewer `<unk>` or broken subwords**, meaning shorter input lengths and lower compute.
* Your model sees more meaningful units during training/inference.

---

#### 🕰 Why Pretrained Tokenizers Don’t Cover Everything

* Pretrained vocabularies (like OpenAI’s `o200k_base`) aim to cover broad general text, but might **split rare jargon or new slang**.
* Any vocabulary has to balance between size and coverage. You can’t include every possible word—especially for specialized datasets.

---

#### ✅ TL;DR

* **Training a custom BPE tokenizer** builds vocab and merge rules from your data—not the model itself.
* Using one optimizes how your model reads and understands domain-specific text.
* Even the best prebuilt tokenizers are general-purpose. If you have specialized needs, a custom tokenizer helps.
* Tokenizers **cover text** based on their trained merges; they don’t automatically include all possible words in a dictionary.

---


[9]: https://huggingface.co/learn/llm-course/en/chapter6/5?utm_source=chatgpt.com "Byte-Pair Encoding tokenization - Hugging Face LLM Course"
[10]: https://huggingface.co/docs/tokenizers/en/quicktour?utm_source=chatgpt.com "Quicktour - Hugging Face"
[11]: https://medium.com/%40shail251298/mastering-tokenizers-part-2-a-comprehensive-guide-to-byte-pair-encoding-bpe-tokenizer-d90236623d6a?utm_source=chatgpt.com "Mastering Tokenization: Part 2 — A Comprehensive Guide to Byte ..."



## Q what kind of data does it expect as raw data? are there any guidelines?

When training a custom BPE tokenizer, the raw data you feed into the training process is plain text—ideally a representative sample of the content your model will handle. Here’s how to prepare it effectively:

📁 1. Supported Raw Data Formats
- Plain text files: .txt files with UTF‑8 encoded text

- Large corpora: collections of documents such as scraped web pages, articles, chat logs, source code, transcripts, etc.

- Cleaned and normalized text: pre-process for consistency (lowercase, unify quotes, strip HTML, handle newlines, etc.) 

The tokenizer library (like Hugging Face tokenizers) reads each line/paragraph and learns merge pairs from the raw byte or word sequences 

🧹 2. Pre‑tokenization & Normalization Guidance
Before BPE training:

- Normalize punctuation and whitespace: replace quotes, multiple spaces, accents, etc. to reduce noise 

- Choose pre-tokenization method:

- Byte-level: start with raw UTF-8 bytes; ideal for web text and multilingual corpora 

- Word-level: split on whitespace, carefully retain punctuation as separate tokens .

Proper normalization ensures your tokenizer encodes text consistently and efficiently.

⚙️ 3. Corpus Size & Resource Considerations
- Training on a large corpus (GBs) requires memory—often 64 GB+ RAM. 

- [discuss.huggingface.co](https://discuss.huggingface.co/t/how-much-memory-is-needed-for-training-bytelevelbpetokenizer/1165)
- https://arxiv.org/html/2402.01035v2

- You can train effectively with a subset—e.g., a few million lines (~200 MB–1 GB). 

- https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

Trade-offs:

- Larger corpora capture more linguistic patterns but use more memory/time.

- Smaller samples are faster and easier to manage but may miss rare tokens.

🎯 4. Vocabulary Size Decisions
Decide your target vocabulary size (typically 30k–100k).

Larger vocab → fewer tokens per text but bigger model embeddings; smaller vocab → more tokens but smaller model.

Studies show:

There's an optimal vocab size based on your model and data domain. 

🧱 5. End-to-End Training Steps
Collect & process your text corpus.

Normalize: fix spacing, punctuation, casing.

Initialize a BPE tokenizer (byte-level or word-based).

Train with Hugging Face:

```
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tok = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=50000, special_tokens=["[UNK]"])
tok.train(["cleaned_corpus.txt"], trainer)

```


Save and use the tokenizer during model training/inference.

This process only builds the vocabulary—it does not train the language model itself.


## adding more training data

When you add more training data later, you have two options:

---

#### 1. **Retrain the tokenizer** 🛠️

* **What it means**: Rerun the BPE algorithm on your combined dataset (original + new) to rebuild the vocabulary and merge rules.
* **Pros**:

  * Captures new patterns and domain-specific terms efficiently.
  * Reduces token counts for unseen or frequent new words—better compression and possibly performance. ([github.com][12], [arxiv.org][13])
* **Cons**:

  * Alters existing token-to-ID mappings, requiring full retraining (or at least embed layer resizing) of any model using it.
  * More computational cost and complexity.

---

#### 2. **Keep the tokenizer fixed**

* **Approach**: Continue using the same tokenizer trained initially.
* **Pros**:

  * Maintains stable token IDs, preserving model compatibility.
  * Simplifies deployment and fine-tuning—no need to update token embeddings or model architecture.
* **Cons**:

  * New words may be split into less efficient subwords or bytes.
  * Slightly larger token counts; potentially minor performance loss on emerging vocabulary.

---

#### 🔄 Can You Incrementally Update?

Most BPE implementations (Hugging Face, tiktoken) don’t support *partial updates* to existing vocabularies—retraining overwrites the old vocabulary ([diva-portal.org][14], [miditok.readthedocs.io][15], [discuss.huggingface.co][16]). You’d either:

* **Merge old + new corpora** and fully retrain, or
* **Extend with new tokens**, but this often changes merge behavior and requires model updates.

---

#### 🧭 Rules of Thumb

* **You don’t have to retrain just because you added data**—stick with your tokenizer unless:

  * The new data has **specialized terms or languages** not well represented before (e.g., legal, biomedical, code). Retraining helps reduce token fragmentation.&#x20;
  * You’re fine-tuning or training a model from scratch—you can afford to rebuild.
* **Keep it fixed when**:

  * You only care about incremental fine-tuning on top of a pre-trained model.
  * Stability and compatibility matter more than minor gains in tokenization efficiency.

---

[12]: https://github.com/huggingface/tokenizers/issues/1109?utm_source=chatgpt.com "How can I keep the initial input vocab and incremental add ... - GitHub"
[13]: https://arxiv.org/html/2402.01035v2?utm_source=chatgpt.com "Getting the most out of your tokenizer for pre-training and domain ..."
[14]: https://www.diva-portal.org/smash/get/diva2%3A1843124/FULLTEXT01.pdf?utm_source=chatgpt.com "[PDF] INCREMENTAL RE-TOKENIZATION IN BPE-TRAINED ... - DiVA portal"
[15]: https://miditok.readthedocs.io/en/latest/train.html?utm_source=chatgpt.com "Training a tokenizer - MidiTok's docs"
[16]: https://discuss.huggingface.co/t/does-tokenizer-changed-during-model-training/19213?utm_source=chatgpt.com "Does tokenizer changed during model training - Hugging Face Forums"


### Q how do we manage context limits with this token length?  what is token length here? is it the total length of all tokens that exist in the trained tokenizer?

Token length refers to the **number of tokens** produced when you encode your text using the model's tokenizer—it isn't about the total vocabulary, but how much of your text fits within the model's context window.

---

## 🔢 What is “token length” and “context limit”?

* **Token length** is the count of tokens (e.g., words, subwords, characters) generated by the tokenizer when encoding your prompt + expected completion.
* **Context limit** (or maximum context length) is the upper bound on combined *prompt* and *response* tokens that a model can process in one call. For instance, GPT‑3.5-turbo allows 4,097 tokens total ([help.openai.com][17], [notes.kodekloud.com][18]).

So if you send a 2,000-token prompt, you can only request up to \~2,097 tokens for the model's generated answer.

---

## ⚙️ How to manage context limits?

1. **Count tokens before sending**
   Use `tiktoken`:

   ```python
   import tiktoken
   enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
   prompt_tokens = len(enc.encode(my_prompt))
   ```

   ([python.langchain.com][19])

2. **Plan for completion**
   Leave room for the model's response:

   ```
   max_completion = max_context - prompt_tokens
   ```

3. **Truncate or chunk long inputs**

   * **Truncate** by dropping excess tokens:

     ```python
     tokens = enc.encode(text)[:max_context]
     ```
   * **Chunk** into multiple segments and process separately ([vellum.ai][20], [cookbook.openai.com][21]).

4. **Implement strategies for long conversations**

   * Rolling context window
   * Summarizing earlier parts
   * Topic-based chunking ([bretcameron.com][22], [reddit.com][23])

---

## ✅ TL;DR

* *Token length* = number of tokens your text becomes.
* *Context limit* = maximum allowed tokens (prompt + reply).
* **Action steps**:

  1. Encode and measure your prompt in tokens.
  2. Reserve space for reply.
  3. If it's too long, truncate or segment your input.

Let me know if you'd like sample code in your preferred language or strategies for conversation memory handling!

[17]: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them?utm_source=chatgpt.com "What are tokens and how to count them? - OpenAI Help Center"
[18]: https://notes.kodekloud.com/docs/Mastering-Generative-AI-with-OpenAI/Understanding-Tokens-and-API-Parameters/What-are-Tokens?utm_source=chatgpt.com "What are Tokens - KodeKloud Notes"
[19]: https://python.langchain.com/docs/how_to/split_by_token/?utm_source=chatgpt.com "How to split text by tokens - Python LangChain"
[20]: https://www.vellum.ai/blog/count-openai-tokens-programmatically-with-tiktoken-and-vellum?utm_source=chatgpt.com "How to Count Tokens with Tiktoken programmatically - Vellum AI"
[21]: https://cookbook.openai.com/examples/embedding_long_inputs?utm_source=chatgpt.com "Embedding texts that are longer than the model's maximum context ..."
[22]: https://www.bretcameron.com/blog/three-strategies-to-overcome-open-ai-token-limits?utm_source=chatgpt.com "3 Strategies to Overcome OpenAI Token Limits - Bret Cameron"
[23]: https://www.reddit.com/r/OpenAI/comments/173lnli/this_models_maximum_context_length_is_4097_tokens/?utm_source=chatgpt.com "\"This model's maximum context length is 4097 tokens\" but tiktoken ..."



### Q why do we have this limit in context length? does it mean that if our vocab size is more, we can get more context length?


The **context length limit** in large language models refers to the **maximum number of tokens** (text, image, or other modality tokens) the model can process in a single forward pass. It’s **not** determined by vocabulary size, but rather by the model’s architecture and design choices.

---

#### 🧠 Why Do We Have Context Length Limits?

1. **Technical Constraints of Transformers**

   * Transformers use a **self-attention mechanism**, which computes relationships between every pair of tokens. This is an $O(N^2)$ computation, where $N$ is the sequence length. Doubling $N$ quadruples the computational cost and memory usage—leading to steep hardware and latency demands ([groq.com][24], [en.wikipedia.org][25]).

2. **Attention Costs**

   * As context length grows, each token’s attention to all earlier tokens becomes slower and more resource-intensive. This limits real-time performance .

3. **Diminishing Returns Beyond a Certain Limit**

   * Studies show that after an optimal window, extra context may not meaningfully improve performance and can even degrade it ([en.wikipedia.org][25]).

4. **Security Concerns**

   * Longer context windows broaden attack surfaces—more text to exploit adversarial prompts or “jailbreaks” ([ibm.com][26]).

---

#### 🤔 Does Vocabulary Size Affect Context Length?

No. **Vocabulary size**—the total number of unique tokens—**does not change** the model’s context window.

* A larger vocabulary can encode more text with fewer tokens, which **indirectly improves** the effective context you can cover—but it doesn’t increase the hard token limit.
* For example, encoding richer words might mean fewer tokens needed to represent the same text, but the model’s **maximum allowable tokens remains fixed** ([reddit.com][27]).

---

#### 🔄 Context vs. Vocabulary: What You Can Control

* **Use a richer tokenization scheme** (like byte-level BPE), which can compress your input into fewer tokens per word, maximizing usable context.
* **But the model’s architecture imposes a cap**—e.g., GPT-4 uses up to 128K tokens, LLaMA 3 supports 128K, and Gemini models go up to **1 million tokens** .

---

#### ✅ TL;DR

* **Context limit** = Max number of tokens model can *process* (due to attention and compute constraints).
* **Vocabulary size** affects *how efficiently you encode text*—but does **not** change the model’s token capacity.
* To maximize context:

  * Choose efficient tokenizers (BPE, byte-level) to reduce token count.
  * Employ strategies like chunking, summarization, or retrieval to handle long inputs.
  * Understand and respect the model’s fixed context window.

Let me know if you'd like help selecting tokenizer settings, performance comparisons, or strategies to handle very long documents efficiently!

[24]: https://groq.com/the-crucial-role-of-context-length-in-large-language-models-for-business-applications/?utm_source=chatgpt.com "The Crucial Role of Context Length in Large Language Models for ..."
[25]: https://en.wikipedia.org/wiki/Transformer_%28deep_learning_architecture%29?utm_source=chatgpt.com "Transformer (deep learning architecture)"
[26]: https://www.ibm.com/think/topics/context-window?utm_source=chatgpt.com "What is a context window? - IBM"
[27]: https://www.reddit.com/r/MachineLearning/comments/198xx6o/d_does_the_vocabulary_size_really_affect_the_size/?utm_source=chatgpt.com "[D] Does the vocabulary size really affect the size of textual LLMs?"


# Word Embeddings

## 📋 What is a vector?

* A vector is a list (tuple) of numbers, often called **scalar values**:

  Each $v_i$ is a real number, and $n$ is the **dimensionality** of the vector ([machinelearningmastery.com][1]).

* In code (like NumPy), that's exactly what it is: a 1D array:

  ```python
  import numpy as np
  v = np.array([1.0, 2.5, -0.3])  # vector of length 3 (3-dimensional)
  ```

  A vector of length n = an n-dimensional vector .

---

### 📏 Why call it a vector?

Because it behaves like a mathematical object in **n-dimensional space**:

* Each number is a **coordinate** along one axis.
* You can do math with it—like add two vectors or scale them by a number.
* It's exactly like a point or arrow in an n-dimensional coordinate system ([stackoverflow.com][2], [math.stackexchange.com][3], [en.wikipedia.org][4]).

---

### 🔢 Comparison: array vs. vector vs. tensor

* **1D array** = vector.
* **2D array** = matrix.
* **ND array (N ≥ 3)** = tensor ([numpy.org][5], [neptune.ai][6]).
* Machine learning uses vectors to represent data, features, embeddings, weights, etc. ([shelf.io][7]).

---
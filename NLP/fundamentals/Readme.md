# Quick Notes

- Sentences can first be split into words (or subword units) called **tokens** using **tokenization**
- These tokens are then assigned an integer value called a **token ID**, which can be converted into a one-hot encoded vector as shown later.
- The process of assigning any non-numeric data, such as images and text, a numerical representation is called **embedding**, and so these vector representations of words are known as **word embeddings**. 
- **Word Embedding** is the process of taking a word and creating a vector representation in N-dimensional space. 
- Prior to 2013, word embeddings were often created using one-hot encoding. This method for producing vector representations is very simple: for each word construct a vector with 0s in every element, except at the position equal to the token ID which should be filled with a 1. This creates a unique vector for each word, where the position of the 1 indicates which word is being encoded (hence the name ‘one-hot’). Because of this, one-hot vectors are called localist representations, as all the information that represents the word is restricted to a single element.
- The collection of words a model can encode is called the **vocabulary**, and number of words in the vocabulary is called the **vocabulary size**.
- **word2vec** is a family of algorithms that produce distributed word embeddings for use in NLP tasks. These vectors are far denser than those created using the one-hot encoding method (i.e. very few, if any, of the elements are 0), and so they can be much smaller in size. The idea is to create an N-dimensional vector space, in which similar words are geographically close to each other. 
- Typically, these embeddings have around 300 dimensions. Once these embeddings are created, they can be written to a file and loaded into memory when needed to essentially form a lookup table at run time. When a language model is given some input text, the text is first converted into tokens. 
- These are then converted into vectors by finding the appropriate row in the word2vec embeddings matrix. For this reason, the embeddings produced by word2vec are called static. These static embeddings form the basis for the so-called dynamic or contextual embeddings that are used in LLMs, which are made by adding context from the surrounding sentences or paragraphs to each word.
- **Static embedding** method predates transformers and suffers from one major drawback: the lack of contextual information. Words with multiple meanings (called polysemous words) are encoded with somewhat ambiguous representations since they lack the context needed for precise meaning. 
- A classic example of a polysemous word is bank. Using a static embedding model, the word bank would be represented in vector space with some degree of similarity to words such as money and deposit and some degree of similarity to words such as river and nature. **This is because the word will occur in many different contexts within the training data. This is the core problem with static embeddings: they do not change based on context — hence the term “static”.**
- The word2vec algorithms process a sentence one word at a time, which the white paper refers to as the center word, denoted w(t). Since word2vec is a distributed model, the algorithms also consider the surrounding context words, called outside words. The number of words considered before and after the center word is determined by a hyperparameter called the window size, which is chosen by the user before training the model. For a window size of  1 , the model will take 1 word before and after the center word to create the list of outside words. These are referred to mathematically as w(t-1) and w(t+1) respectively.
- A **key difference between static and learned embeddings** is the way in which they are trained. Static embeddings are trained in a separate neural network (using the Skip-Gram or Continuous Bag of Words architectures) using a word prediction task within a given window size. Once trained, the embeddings are then extracted and used with a range of different language models. Learned embeddings, however, are integral to the transformer you are using and are stored as weights in the first linear layer of the model. These weights, and consequently the learned embedding for each token in the vocabulary, are trained in the same backpropagation steps as the rest of the model parameters.
- The original papers propose two methods for learning these word vectors called **Skip-Gram and Continuous Bag of Words (CBOW)** These methods are very similar to each other, with both using a neural network with a single hidden layer to generate the word vectors. The difference lies in their objectives:
  - Skip-Gram: Takes in a center word and predicts the outside words
  - CBOW: Takes in some outside words and predicts the center word
- The static embedding method predates transformers and suffers from one major drawback: the lack of contextual information. Words with multiple meanings (called polysemous words) are encoded with somewhat ambiguous representations since they lack the context needed for precise meaning.
- **Transformers overcome the limitations of static embeddings by producing their own context-aware transformer embeddings. In this approach, fixed word embeddings are augmented with positional information (where the words occur in the input text) and contextual information (how the words are used). These two steps take place in distinct components in transformers, namely the positional encoder and the self-attention blocks, respectively.**
- By incorporating this additional information, transformers can produce much more powerful vector representations of words based on their usage in the input sequence. Extending the vector representations beyond static embeddings is what enables Transformer-based models to handle polysemous words and gain a deeper understanding of language compared to previous models.
  - **Static embeddings**: Each word (or token) in the vocabulary has exactly one fixed vector representation. No matter where or how the word appears, its embedding is the same.
  - **Learned embeddings (contextual embeddings)**: The embedding of a token depends on its context (the surrounding words, sentence, etc.). Even the initial token embeddings themselves are parameters inside a larger model (e.g. a Transformer) that are updated as part of training that whole model.
- Learned embeddings are parameters (weights) in the model. During training, you update them via backpropagation along with all the other model weights. 
- After training is completed (or once you decide not to update them anymore), those embedding weights don’t change. They’re fixed. So they provide a fixed “starting point” for each token’s vector representation whenever you input that token.
- Then, for each input during inference, you still add positional encoding and apply the Transformer layers to compute context / usage-dependent representations. But the raw embedding lookup always yields the same vector for that token (before adding position and context).
- If you fine-tune the model on new data, then the embedding weights do change again. So “never change” is conditional on “no fine-tuning / freezing embeddings / during inference”.
- Sometimes in transfer learning you might freeze embeddings (so they don’t change) during some phase of training, then unfreeze them, or allow small updates. So “never change” may not apply throughout all uses.
- **The functions used to generate positional information must produce values that are**:
  - Bounded — values should not explode in the positive or negative direction but be constrained (e.g. between 0 and 1, -1 and 1, etc)
  - Periodic — the function should produce a repeating pattern that the model can learn to recognise and discern position from
  - Predictable — positional information should be generated in such a way that the model can understand the position of words in sequence lengths it was not trained on. For example, even if the model has not seen a sequence length of exactly 412 tokens in its training, the transformer should be able to understand the position of each of the embeddings in the sequence.
- **These constraints ensure that the positional encoder produces positional information that allows words to attend to (gain context from) any other important word, regardless of their relative positions in the sequence.** 
- In theory, with a sufficiently powerful computer, words should be able to gain context from every relevant word in an infinitely long input sequence. The length of a sequence from which a model can derive context is called the context length. In chatbots like ChatGPT, the context includes the current prompt as well as all previous prompts and responses in the conversation (within the context length limit). This limit is typically in the range of a few thousand tokens, with GPT-3 supporting up to 4096 tokens and GPT-4 enterprise edition capping at around 128,000 tokens - **The goal of self-attention is to move the embedding for each token to a region of vector space that better represents the context of its use in the input sequence.**
- The “Attention is All You Need” paper extends standard self-attention into Multi-Head Attention (MHA) by dividing the attention mechanism into multiple heads. In standard self-attention, the model learns a single set of weight matrices (W_Q, W_K, and W_V) that transform the token embedding matrix X into query, key, and value matrices (Q, K, and V). These matrices are then used to compute attention scores and update X with contextual information as we have seen above.
- In contrast, MHA splits the attention mechanism into H independent heads, each learning its own smaller set of weight matrices. These weights are used to calculate a set of smaller, head-specific query, key, and value matrices (denoted Q^h, K^h, and V^h). Each head processes the input sequence independently, generating distinct attention outputs. These outputs are then concatenated (stacked on top of each other) and passed through a final linear layer to produce the updated X matrix, shown as Y in the diagram below, with rich contextual information.
- By introducing multiple heads, MHA increases the number of learnable parameters in the attention process, enabling the model to capture more complex relationships within the data. Each head learns its own weight matrices, allowing them to focus on different aspects of the input such as long-range dependencies (relationships between distant words), short-range dependencies (relationships between nearby words), grammatical syntax, etc. The overall effect produces a model with a more nuanced understanding of the input sequence.

![MHA](/assets/MHA.webp)

- Decoder-Only Models:

  - Goal: Predict a new output sequence in response to an input sequence
  - Overview: The decoder block in the Transformer is responsible for generating an output sequence based on the input provided to the encoder. Decoder-only models are constructed by omitting the encoder block entirely and stacking multiple decoders together in a single model. These models accept prompts as inputs and generate responses by predicting the next most probable word (or more specifically, token) one at a time in a task known as Next Token Prediction (NTP). As a result, decoder-only models excel in NLG tasks such as: conversational chatbots, machine translation, and code generation. These kinds of models are likely the most familiar to the general public due to the widespread use of ChatGPT which is powered by decoder-only models (GPT-3.5 and GPT-4).

- Encoder-Only Models:

  - Goal: Make predictions about words within an input sequence
  - Overview: The encoder block in the Transformer is responsible for accepting an input sequence, and creating rich, numeric vector representations for each word (or more specifically, each token). Encoder-only models omit the decoder and stack multiple Transformer encoders to produce a single model. These models do not accept prompts as such, but rather an input sequence for a prediction to be made upon (e.g. predicting a missing word within the sequence). Encoder-only models lack the decoder used to generate new words, and so are not used for chatbot applications in the way that GPT is used. Instead, encoder-only models are most often used for NLU tasks such as: Named Entity Recognition (NER) and sentiment analysis. The rich vector representations created by the encoder blocks are what give BERT a deep understanding of the input text. The BERT authors argued that this architectural choice would improve BERT’s performance compared to GPT, specifically writing that decoder-only architectures are: “sub-optimal for sentence-level tasks, and could be very harmful when applying finetuning based approaches to token-level tasks such as question answering” [1]

![enc_dec](/assets/enc_dec.webp)

- GPT influenced the development of BERT in several ways. Not only was the model the first decoder-only Transformer derivative, but GPT also popularised model pre-training. Pre-training involves training a single large model to acquire a broad understanding of language (encompassing aspects such as word usage and grammatical patterns) in order to produce a task-agnostic foundational model. In the diagrams above, the foundational model is made up of the components below the linear layer (shown in purple). Once trained, copies of this foundational model can be fine-tuned to address specific tasks. Fine-tuning involves training only the linear layer: a small feedforward neural network, often called a classification head or just a head. The weights and biases in the remainder of the model (that is, the foundational portion) remained unchanged, or frozen.

- **Bidirectionality is perhaps BERT’s most significant feature and is pivotal to its high performance in NLU tasks, as well as being the driving reason behind the model’s encoder-only architecture**. While the self-attention mechanism of Transformer encoders calculates bidirectional context, the same cannot be said for decoders which produce unidirectional context. The BERT authors argued that this lack of bidirectionality in GPT prevents it from achieving the same depth of language representation as BERT.
- Bidirectional denotes that each word in the input sequence can gain context from both preceding and succeeding words (called the left context and right context respectively). In technical terms, we say that the attention mechanism can attend to the preceding and subsequent tokens for each word. To break this down, recall that BERT only makes predictions about words within an input sequence, and does not generate new sequences like GPT. Therefore, when BERT predicts a word within the input sequence, it can incorporate contextual clues from all the surrounding words. This gives context in both directions, helping BERT to make more informed predictions.
- Contrast this with decoder-only models like GPT, where the objective is to predict new words one at a time to generate an output sequence. Each predicted word can only leverage the context provided by preceding words (left context) as the subsequent words (right context) have not yet been generated. Therefore, these models are called unidirectional.

![bidir_context](/assets/bidir_context.webp)

- Training a bidirectional model requires tasks that allow both the left and right context to be used in making predictions. Therefore, the authors carefully constructed two pre-training objectives to build up BERT’s understanding of language. These were: the Masked Language Model task (MLM), and the Next Sentence Prediction task (NSP). 
- The **Masked Language Modelling task** was created to directly address the need for training a bidirectional model. To do so, the model must be trained to use both the left context and right context of an input sequence to make a prediction. This is **achieved by randomly masking 15% of the words in the training data**, and training BERT to predict the missing word. In the input sequence, the masked word is replaced with the [MASK] token. For example, consider that the sentence _A man was fishing on the river exists_ in the raw training data found in the book corpus. When converting the raw text into training data for the MLM task, the word fishing might be randomly masked and replaced with the [MASK] token, giving the training input _A man was [MASK] on the river_ with target _fishing_. Therefore, the goal of BERT is to predict the single missing word fishing, and not regenerate the input sequence with the missing word filled in. The masking process can be repeated for all the possible input sequences (e.g. sentences) when building up the training data for the MLM task. This task had existed previously in linguistics literature, and is referred to as the Cloze task [8]. However, in machine learning contexts, it is commonly referred to as MLM due to the popularity of BERT.
- BERT's Special tokens
  - [PAD] (token ID: 0) — a padding token used to bring the total number of tokens in an input sequence up to 512.
  - [UNK] (token ID: 100) — an unknown token, used to represent a token that is not in BERT’s vocabulary.
  - [CLS] (token ID: 101) — a classification token, one is expected at the beginning of every sequence, whether it is used or not. This token encapsulates the class information for classification tasks, and can be thought of as an aggregate sequence representation.
  - [SEP] (token ID: 102) — a separator token used to distinguish between two segments in a single input sequence (for example, in Next Sentence Prediction). At least one [SEP] token is expected per input sequence, with a maximum of two.
  - [MASK] (token ID: 103) — a mask token used to train BERT on the Masked Language Modelling task, or to perform infere_nce on a masked sequence.
- **Base** refers to the standard version of a model that can be fine-tuned on a downstream task, while **chat and instruct** refer to specific fine-tuned versions of base models that have been trained for chatbot and instruction tasks respectively. Chat models are fine-tuned on conversation data, and are designed for conversational chatbot applications such as virtual assistants and ChatGPT-style use-cases. Instruct models on the other hand are designed to receive instructions and respond to them.
- Mistral AI Architecture:
  - Mistral AI have utilised a number of these advancements to improve the efficiency of Mistral 7B, achieving a highly performant model with a fraction of the parameters. In the following sections we will explore these advancements, which include:

    - RMS Normalization — replacing Layer Normalization
    - Rotary Position Embedding (RoPE) — replacing Absolute Positional Encoding
    - Grouped Query Attention (GQA) — replacing Multi-Head Attention
    - Sliding Window Attention (SWA) — improving training and inference speed, particularly for long sequences
    - Rolling Buffer KV Cache — improving training and inference speed, in conjunction with SWA
    - SwiGLU Activation Function — replacing ReLU in the Feed Forward sub-layers

![mistral](/assets/mistral.webp)

- Normalization Sub-Layer: Normalization is required in Transformer-based models due to an issue known as covariate shift. This describes the phenomenon in which some weights in a model receive significant updates while others do not. This change in distribution of the weights can have a knock-on effect in the next layer of the network, causing further unstable updates to weights during backpropagation and a drop in performance. Normalization standardises the inputs to each layer by ensuring a consistent mean and variance across the input vector, which in turn stabilises the learning process.

- Feed Forward Sub-Layer: The Feed Forward step introduces non-linear transformations and additional learning capacity. In simple terms, these components allow the model to determine how best to improve its own internal representations of text by learning from training data. Feed Forward blocks are shallow neural networks consisting of: an input layer, one hidden layer, and an output layer. In the Transformer, the inputs to the Feed Forward network are the outputs from the Normalization step (we will see later that this is slightly different for Mistral 7B). The Feed Forward network takes in these numerical representations of the input sequence and updates them in a way that helps the model produce a good output sequence. By using a neural network approach, we eliminate the need to impose strict rules on how the model must augment these representations and instead allow the model to learn how best to change them via backpropagation.


- Example:

  For a more concrete example, consider how the original Transformer processes the input sequence: “Write a poem about a man fishing on a river bank”.

  1. **Tokenization**: Divide the input sequence into the tokens write, a, poem, about, a, man, fishing, on, a, river, and bank. For more about tokenization, see Part 1 of this series.

  2. **Embedding**: Map each token to its corresponding learned embedding. These are vector representations of the tokens which encode their general meaning. For more about embeddings, see Part 2 of this series.

  3. **Multi-Head Attention**: Pass the embeddings into the Attention block to update the vector representation of each word with contextual information. This ensures that words such as bank are given more appropriate vector representations depending on their usage (e.g. river bank, monetary bank, etc.). For more about Attention blocks, see Part 3 of this series.

  4. **Normalization**: Pass the contextual embeddings from the Attention block to the Normalization block. Here, the vectors of inputs are normalized to ensure a consistent mean and variance, mitigating the problem of covariate shift.

  5. **Feed Forward**: Pass the output from the Normalization step to the Feed Forward sub-layer. This step updates the vector representation for each token in such a way that helps the model produce a nice poem later in the process. The specific steps for updating the vector representations are not hard-coded but rather learned by the model via backpropagation.

  6. **Normalization**: Pass the outputs of the Feed Forward step to another Normalization block. Steps 3–6 repeat N times (where N is the number of encoder blocks) before the vector representations are sent to the decoder block.

- The Need for Residual Connections:

The architecture diagram below shows that inputs to the Attention and Feed Forward sub-layers are passed to the Normalization sub-layers via residual connections (highlighted in red). These inputs are added to the Attention and Feed Forward outputs respectively before normalization, hence the “Add” in the “Add & Norm” label. Residual connections help address an issue known as the vanishing gradient problem, a common challenge in training deep neural networks. During backpropagation, gradients (partial derivatives of the loss function with respect to each weight) determine the direction and magnitude of weight updates. However, these gradients can sometimes become extremely small as they propagate through many layers, leading to negligible changes in some weights. This can cause earlier layers in the network to learn very slowly as their gradients approach zero. Residual connections alleviate this problem by allowing gradients to flow more directly to earlier layers, bypassing some intermediate layers. This additional pathway helps maintain gradient strength, ensuring stable updates and preventing the model from “forgetting” what it has learned in earlier layers. In short, including a residual connection at each Normalization stage provides an additional path for backpropagated gradients and prevents the model from learning slowly in its earlier layers.

![residual](/assets/residual.webp)

- Visualising LayerNorm
  - **LayerNorm** transforms the distribution of inputs to a network such that the values follow a Gaussian distribution. Consider the example shown in the image below, which focuses on Normalization directly after the Attention step. Here, the input to LayerNorm will be the sum of the Attention inputs and Attention outputs, the result of which is a matrix of contextual token embeddings for each token in the input sequence (in this case, “Write a poem about a man fishing on a river bank”). The dimensions of this matrix are L_max x d_model, where L_max is the input sequence length and d_model is the number of embedding dimensions. The columns of this matrix store the token embedding for each token of the input sequence. For example, the first column stores the contextual embedding for “write”, the second for “a”, and so on.

  - A frequency plot using a histogram can be drawn to approximate the distribution of values for each individual token embedding. The image below shows an example with the embedding for “bank.” Before normalization, the values in the embedding vector for “bank” have a mean of 18.5, whereas afterwards, the mean is reduced to 0. The normalization process is applied to each column of the matrix separately, with each normalized according to its own mean and variance.

![layer_norm](/assets/layer_norm.webp)

- To normalize the token embeddings, we first calculate two key statistical values for each column: mean and variance. These values describe the centre and spread of the data, respectively. Once these have been established, each value in the input vector can be adjusted according to the normalization formula. Let’s briefly break down these formulae:

  - **Mean**: The mean (average) describes the centre of a distribution and is calculated by summing all the values in a column and dividing by the number of values (dimensions) in the column.
  - **Variance**: The variance describes the amount of spread (variation) within a distribution and is given by the average squared distance between each data point and the mean. A higher variance indicates that the data points are more spread out, while a lower variance indicates that the values cluster around the mean. The use of squared differences rather than absolute differences is partly due to historical reasons but also because it provides a differentiable measure of spread. This is a property that comes in very useful in advanced statistics, and so variance has become a standard measure in the field.
  - **Normalization**: The normalization process involves two main formulae. The first (shown on the left of the two in the image below) transforms the column’s current distribution into a Normal distribution. This works by subtracting the mean from each value in the column so that the distribution is centred at 0, and then dividing by the square root of the variance (called the standard deviation). This division ensures the standard deviation of the resulting distribution is 1, which is a requirement for the Normal distribution. An additional term, ϵ, is included to prevent division by 0 when there is no spread in the data. The second formula applies learnable adjustments to these normalized values using two parameters: a scale factor, γ, and an offset, β. These parameters are learned by the model during training through backpropagation. The γ and β values are specific to each feature (row in the matrix), not to each embedding (column). Therefore, each dimension of an embedding will undergo a transformation using distinct γ and β values. This allows the model to learn flexible transformations within the embedding space, improving its ability to represent complex patterns in the data.

![mean_variance](/assets/mean_variance.webp)

- Mistral 7B uses an improvement to LayerNorm called **Root Mean Square Normalization, or RMS Norm**, introduced by Zhang and Sennrich in 2019 [7]. The authors hypothesised that the effectiveness of LayerNorm was due to rescaling the values (dividing by the variance) and not so much recentering them (subtracting the mean).

- Therefore, if the calculation of the mean could be omitted, the model would see a significant speed boost during the training phase. The issue here however, is that the calculation of the variance itself also requires the mean to be known. Hence, the authors set out to identify a new rescaling method that would become RMS Normalization.

- Rotary Position Embedding (RoPE):
  - Unlike older architectures (such as Recurrent Neural Networks), Transformer-based models process all of their input tokens in parallel, not sequentially. While this parallel processing improves speed, it also results in a loss of positional information since the tokens are not processed in order. Therefore, some form of positional encoding is needed to inject this information back into the embedding vectors, and this can be achieved in various ways.

  - Absolute Positional Encoding: The sinusoidal positional encoding technique introduced in the original Transformer uses sine and cosine functions to create a positional encoding vector for each token in the input sequence. These vectors are then added to the learned embeddings via vector addition. The positional encodings depend solely on the absolute position of the tokens in the sequence and do not change based on the input sequence itself. For example, the token at position 0 will always have the same positional encoding, regardless of the sequence. Hence, this method is called absolute positional encoding.

  - One limitation of this approach is that it only represents the absolute position of tokens, not their relative distances. For instance, the distance between the tokens in positions 3 and 5 of a sequence versus 103 and 105 is identical, but this information is not captured with absolute positional encoding. Intuitively, tokens that are closer together are likely to be more relevant than those that are further apart, and encoding this information about relative positioning could significantly improve model performance.

  - Relative Positional Encoding: In April 2018, researchers at Google (including two authors of the original Transformer paper) published “Self-Attention with Relative Position Representations”, a paper that outlined a new paradigm for positional encoding [8]. The authors explored the use of relative positional encoding, which captures information about the relative distance between tokens as well as their absolute positions. For example, in the sentence “Write a poem about a man fishing on a river bank”, the words “poem” and “man” are three words apart, in the same way that “on” and “bank” are three words apart. This type of positional encoding has been used in prominent models such as Dai et al.’s Transformer-XL (2019) [9] and Google’s T5 (2020) [10].

  - Although relative positional encoding improves a model’s ability to capture the relationship between tokens, it significantly increases the training time. As models grow larger, adding components that increase training time becomes less practical. Additionally, challenges like integrating an KV cache (which we will cover later in this article) have caused many researchers to move away from this technique. We will not cover the details of the original relative positional encoding technique, but if you are interested, I highly encourage you to read through the paper.

  - Rotary Position Embeddings (RoPE): Rotary embeddings were introduced by Su et al. in their 2020 paper “RoFormer: Enhanced Transformer with Rotary Position Embedding”, and offer a unique approach to encoding positional information [11]. Unlike sinusoidal encoding, which adds positional information directly to the token embeddings, rotary embeddings instead apply a rotation to the query and key vectors for each token. The rotation angle for each token is based on its absolute position in the sequence. For example, in the input “write a poem about a man fishing on a river bank”, the query and key vectors for poem (at position 2) are rotated by 2θ, while the query and key vectors for man (at position 5) are rotated by 5θ, and so on. Note that token position is zero-indexed, meaning we start counting at 0 instead of 1 (therefore write is said to be at position 0 and so its query and key vectors are not rotated). This approach captures not only the absolute position of the token but also the relative positions, since man and poem are 3θ apart, which represents a distance of 3 tokens.

  - Encoding positional information with angular displacement also offers a few nice properties that work well with existing transformer components. For example, the self-attention mechanism relies heavily on the dot-product operation, which already considers the angular distance between queries and keys in its formulae. Additionally, the angular distance between two tokens remains unchanged if more tokens are added before or after them. This allows for modifications to the input sequence without significantly altering the positional information, unlike the absolute positional encoding method.

  - Implementing RoPET he outline above gives a simplified overview of RoPE to illustrate its core concepts, but the technical implementation includes two important details:

    1. Pair-wise feature rotation: The features of each query/key vector are rotated in pairs within the embedding space.

    2. Multi-frequency positional encoding: Each feature pair in a query/key vector is rotated by a slightly different angle.

    Let’s look at how RoPE integrates into transformer-based architectures, the mathematics behind its implementation, and understand what the two details above mean and why they are needed for RoPE to function effectively.

  — Integrating RoPE into Transformers: Transformers using RoPE process text with the following steps:

    1. Tokenization and Embedding: As always, the process begins when a model receives an input sequence which is tokenized to produce a list of token IDs. These token IDs are then transformed into token embeddings, creating a matrix where each column corresponds to the embedding vector of a single token.

    2. Normalization: In the original Transformer model, positional information is added directly to the raw token embeddings at this stage. However, in models using RoPE, the token embeddings are first normalized. This step stabilises training by preventing covariate shift, as discussed earlier (see the architecture diagram in Section 2.1).

    3. Calculate Query, Key, and Value Matrices: The model then calculates the Query, Key, and Value matrices (Q, K, and V) needed for the attention mechanism. This is achieved by multiplying the normalized embeddings matrix by the corresponding weight matrices, W_Q, W_K, and W_V. Here, the columns of the resulting matrices represent the query, key, and value vectors for each token respectively. The Query and Key matrices are used to compute attention scores, which then weight the values in the Value matrix to produce context-aware outputs in the attention block.
    4. Rotate the Query and Key Matrices: The Query and Key matrices are rotated to incorporate positional information. Since only the Query and Key matrices are involved in calculating attention scores, positional information is added solely to these matrices. As a result, the Value matrix is not rotated. After the attention scores are computed, the Value matrix simply provides the embeddings that will be updated based on the scores. This is why the positional encoding symbol is omitted from the Value matrix in the architecture diagram.

  - Grouped Query Attention (GQA):  Mistral 7B uses Grouped Query Attention (GQA), which itself builds upon Multi-Query Attention (MQA).

<READ MORE FROM ARTICLE>

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

# Attention is all you need

## how can we make the context length more? can we keep on increasing the heads and that should work?


Increasing the context length of transformer-style models (i.e. how many tokens you can feed the model and have it attend over) is an active area of research. No, simply increasing the number of attention heads is *not* sufficient (or efficient) by itself. There are fundamental and practical limitations. Below are **why it's not enough**, the constraints, and the ways people are actually pushing context lengths higher (with trade-offs).

---

### Why more heads alone don’t solve longer context

If you increase the number of heads (say you double them, or go from 8 → 16 → 32 heads) while keeping everything else (hidden size, layers, etc.) more or less the same, you’ll run into several issues:

1. **Compute and memory scale**:

   * Self-attention cost is quadratic in sequence length *n* (i.e. O(n²·d\_model) roughly) because each of n query tokens attends to n key tokens. Doubling heads doesn’t reduce that O(n²) factor. You still have to compute pairwise attention scores.
   * Also, you need memory to store attention score matrices (which is size n×n), key/value caches etc. That blows up with sequence length.

2. **Dimensionality per head shrinks**:

   * If you keep `d_model` fixed (total hidden dimensionality), then increasing number of heads splits the hidden size per head thinner. If each head becomes too small, each head’s capability (feature space) becomes limited, which can reduce effectiveness.

3. **Positional encodings / generalization**:

   * Many transformers are trained with a fixed maximum context length (e.g. 2048, 4096 tokens). Positional encodings, relative or absolute, are set up (and tuned) for that window. Going far beyond that without retraining or special handling tends to degrade performance.

4. **Training stability & data**:

   * To make models understand long context, you need enough training data with long sequences. If the model is never exposed to sequences of say length 100k, at inference time it will struggle.
   * Also gradients, caching, hardware constraints (GPU/TPU memory, bandwidth) often limit how long sequences can be during training.

---

### Techniques that *are* used to extend context length

Because of the constraints above, researchers have developed several methods for extending context length more efficiently. Some of these include:

| Method                                                 | Basic idea                                                                                                                                                                                                                                       | Trade-offs / when it's used                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Sparse attention / local windows / sliding windows** | Limit each token to attend only to a subset of other tokens (e.g. tokens within a local window, or selected key tokens) rather than all. This reduces cost and memory. Example: Routing Transformer; “Efficient Content-Based Sparse Attention”. | Less ability to attend to far away tokens in certain layers; may need multi-hop patterns or special routing to get global context. |
| **Relative positional encodings, interpolation**       | Use position encodings that generalize, or fine-tune to extend positional embeddings beyond originally trained max length. E.g. “Position Interpolation” for RoPE (rotary position embeddings).                                                  | Needs additional fine-tuning; may have diminishing performance the further you extrapolate.                                        |
| **Hierarchical approaches / summarization / chunking** | Break the input context into chunks, optionally summarize / compress (via special tokens or “landmark” tokens) and then allow the model to attend over summaries + some detailed chunks.                                                         | Loss of detail in summarized parts; overhead of extra summarization steps; complexity adds.                                        |
| **Memory / recurrence approaches**                     | Use cached past hidden states, cross-chunk attention (e.g. Transformer-XL, Compressive Transformer) to allow the model to remember previous chunks without recomputing them.                                                                     | Requires careful memory management; may accumulate errors; might still limit how far back the model has “fresh” detailed context.  |
| **Efficient / IO-aware attention implementations**     | Methods like FlashAttention that optimize how attention is computed (tiling, better memory access) so that longer contexts are more feasible / efficient.                                                                                        | Doesn’t fundamentally change O(n²) costs, but reduces constants / makes bigger n more practical.                                   |
| **Novel attention approximations (SparseK, etc.)**     | Select a subset of key/value pairs per query (e.g. top-k, sparse kernel methods), achieve more favorable scaling (possibly near linear) for many queries.                                                                                        | Approximation may lose precision; sometimes needs fine-tuning; may perform differently for different tasks.                        |

---

### What a real increase in context window entails

To significantly increase context length (say from 4k → 128k tokens or more), you’d likely need a combination of:

* **Increasing d\_model / hidden size / layers** to support more capacity.
* **Using efficient / sparse attention mechanisms** or approximations so you can afford the computation.
* **Proper positional encoding scheme** that generalizes beyond training window (learned, relative, or interpolated).
* **Training with longer sequences / curriculum**: gradually increasing context during pretraining so the model sees longer inputs.
* **Memory strategies** to manage key/value storage, caching, hardware utilization.

---

### Conclusion

* **No**, merely increasing the number of heads doesn’t solve the context length issue by itself. It can help sort of by distributing attention, but it doesn’t reduce the fundamental cost or memory blow-up of attention over very long sequences.
* To make context windows much larger, you need additional techniques (sparse/approximated attention, chunking, better positional encodings, efficient implementations, etc.).

If you like, I can pull up a list of specific architectures or models currently using 100k+ token contexts and detail *how* they accomplish that, so you can see what trade-offs are made.



---


## Encoder-only vs decoder-only vs encoder-decoder(seq2seq)

Here’s a clear explanation of **encoder vs decoder models** in transformer architectures: what each does, how they differ, and which kinds of tasks each is good for.

---

### Basic definitions

* **Transformer** architecture (from *“Attention Is All You Need”*) has two parts: an **encoder** and a **decoder**. The encoder processes input data and builds representations; the decoder uses those representations to produce output, often token by token.
* Over the years, many variants have emerged: *encoder-only*, *decoder-only*, *encoder-decoder (seq2seq)*. Each is a subset or re-use of these blocks. ([Auroria][28])

---

### Encoder-Only Models

#### What they are

* These models consist **only** of the encoder stack (one or more transformer encoder layers). ([Auroria][28])
* They use *self-attention* (bidirectional / full attention over all tokens in the input) so each token’s representation sees all the other tokens in the input. No masking of future positions. ([Vinija][29])

#### Training objectives

* Often trained with *masked language modeling* (MLM), where some tokens in the input are masked out and the model must predict them based on the rest. BERT is a famous example. ([MachineLearningMastery.com][30])
* They produce contextual embeddings of input text. Good for classification, classification of tokens (NER), information extraction, semantic search, etc. ([IBM][31])

#### Pros & Cons

| Pros                                                                                   | Cons                                                                                                                              |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Good at understanding context (you can look both forwards and backwards in the input). | Cannot generate sequence outputs by itself (no auto-regressive generation) unless you add a separate decoding head.               |
| Efficient / simpler for tasks that just need understanding (e.g. classification).      | Not useful for tasks where output is a sequence conditioned on some input (like translation, summarization) without modification. |

---

### Decoder-Only Models

#### What they are

* These are models made up **only** of decoder layers. They generally use *causal / masked self-attention*, meaning each token can only attend to previous tokens, not future ones. This ensures the model predicts next token(s) in a sequence in a valid order. ([Auroria][28])
* There is *no encoder stack*. All inputs (prompts, context) are fed into this stack as tokens, and the model generates output autoregressively. GPT-family models are classic examples. ([Auroria][28])

#### Training objective

* Trained usually with *causal language modeling (CLM)*: predict each next token given all previous tokens. E.g. given “The cat sat on the”, predict “mat”. ([Auroria][28])
* During inference, generation is done token by token, because each next output depends on past outputs.

#### Pros & Cons

| Pros                                                                                                             | Cons                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Very good for generative tasks (text generation, completion, dialogue, story writing).                           | They can be less efficient or less accurate in tasks needing strong contextual understanding of the full input both forward & backward. Because they don’t have full bidirectional attention.        |
| Simpler architecture (no cross-attention / encoder-decoder connections). Might be more straightforward to scale. | Can produce outputs that are less precise for tasks like translation or summarization unless carefully trained / prompt engineered. Also need strategies to avoid generating bad/unwanted sequences. |

---

### Encoder-Decoder (Seq2Seq) Models

#### What they are

* These combine both encoder and decoder stacks. The **encoder** processes the input (e.g. a sentence in the source language) and produces a representation. The **decoder** generates output (e.g. translation) **while attending to both**: past outputs (via masked self-attention) + the encoder output (via *cross-attention*) at each step. ([Auroria][28])
* Common examples: *T5*, *BART*, and the original transformer used for machine translation. ([Wikipedia][32])

#### Pros & Cons

| Pros                                                                                                                                                                            | Cons                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Very well suited for *conditional generation*: you want to generate an output sequence based on some input sequence (translation, summarization, question → answer, etc.).      | More complex architecture → more parameters, more compute. Also needs appropriate training data with input‐output pairs.   |
| Encoder can fully process the input (bidirectional) to build rich representations; decoder uses that to produce output. This often yields higher quality in many seq2seq tasks. | Inference can be slower because you need both encoder + decoder; the decoder works autoregressively (one token at a time). |

---

### When to use which architecture

Here are some guidelines / scenarios:

| Task Type                                                                          | Good architecture(s) |
| ---------------------------------------------------------------------------------- | -------------------- |
| Text classification / sentiment / NER / embedding for search                       | Encoder-Only         |
| Text generation / completion / chatbots / code generation                          | Decoder-Only         |
| Translation / summarization / tasks where output is a transformed version of input | Encoder-Decoder      |

---

### Key Differences in Detail

* **Attention Patterns**:
    Encoder uses full attention over all input tokens (bidirectional).
    Decoder uses *masked self-attention* so tokens only see earlier outputs (for generation), plus *cross-attention* when combined with encoder in seq2seq. ([next.gr][33])

* **Training vs Inference behavior**:
    Encoder-decoder models are usually trained with teacher forcing (decoder gets ground-truth previous token) but inference uses model’s own predictions. Decoder-only models always train in a causal next-token prediction way. ([Auroria][1])

* **Efficiency trade-offs**:
    Encoder-only are cheaper if you only need understanding tasks.
    Encoder-decoder are heavier because you have two stacks and cross-attention etc.
    Decoder-only is simpler in architecture but for long generated outputs it can be slow because each token is generated sequentially.

---


[28]: https://www.auroria.io/the-transformer-architecture/?utm_source=chatgpt.com "The Transformer Architecture"
[29]: https://vinija.ai/models/Transformers/?utm_source=chatgpt.com "Vinija's Notes • Models • Transformers"
[30]: https://machinelearningmastery.com/encoders-and-decoders-in-transformer-models/?utm_source=chatgpt.com "Encoders and Decoders in Transformer Models - MachineLearningMastery.com"
[31]: https://www.ibm.com/think/topics/encoder-decoder-model?utm_source=chatgpt.com "What is an encoder-decoder model? | IBM"
[32]: https://en.wikipedia.org/wiki/T5_%28language_model%29?utm_source=chatgpt.com "T5 (language model)"
[33]: https://www.next.gr/ai/sentiment-analysis/decoder-vs-encoder-in-transformer-models?utm_source=chatgpt.com "Decoder vs Encoder in Transformer Models | AI Tutorial | Next Electronics"


## Feed forward Networks(FFN)

Here’s a detailed explanation of the **feed-forward layer** (also called FFN—Feed-Forward Network) in Transformer architectures: what it is, how it works, why it’s there, and what trade-offs it brings.

---

### What is the Feed-Forward Layer (FFN) in Transformers

The FFN is a sub-component inside each Transformer *block* (both encoder blocks and decoder blocks). It usually comes *after* the multi-head attention sublayer and adds extra transformation + nonlinearity. The sequence is:

```
… → Multi-Head Attention → Add & Normalize → Feed-Forward → Add & Normalize → …
```

So the FFN operates *within* each layer of the Transformer.

---

### Structure & Formula

Typically, the FFN has this form:

$$
\mathrm{FFN}(x) = W_2\big(\phi(W_1 x + b_1)\big) + b_2
$$

Where:

* $x$ is the input vector (for one position in the sequence; as a full batch it’s something like shape `[batch_size, seq_len, d_model]`).
* $W_1$ is a weight matrix that increases dimensionality: from `d_model` to a larger dimension `d_ff` (sometimes called “intermediate size”, “filter size”, etc.).
* $b_1$ is a bias term for that first linear transformation.
* $\phi(\cdot)$ is a non-linear activation (original Transformer uses `ReLU`, many variants use e.g. `GeLU`). ([Wikipedia][34])
* $W_2$ then projects back from `d_ff` down to `d_model`. $b_2$ is the corresponding bias.

So the flow is: **expand → non-linear transform → contract**.

---

### Position-wise Application

An important detail is that this FFN is *position-wise*, meaning:

* It is applied *independently* at each position in the input sequence (and for each token) in parallel.
* There are no interactions across sequence positions in the FFN part. The interactions (mixing information between positions) are handled by the attention mechanism. ([Divya's Blog][35])

So if your input is of shape `[batch_size, seq_len, d_model]`, the FFN applies the same two linear layers + activation to each of the `seq_len` positions separately (though implemented in parallel via a batch matrix multiplication).

---

### Hyperparameters / Common Values

* **`d_model`** is the input / output dimension for one position (the same dimension as embeddings / hidden states).
* **`d_ff`** (intermediate / hidden size of the FFN) is larger, often 4× `d_model` in many implementations. E.g. if `d_model = 512`, then `d_ff = 2048`. ([Wikipedia][34])
* The activation is often ReLU in the original, with dropout between the two linear layers. ([Towards Data Science][36])

---

### Why Do We Need FFN?

Here’s what the FFN brings to the Transformer architecture:

1. **Non-linearity / Feature transformation**
   Attention lets the model mix information across positions, but that mixing is (apart from the softmax) linear in many parts. Nonlinear transformations (via ReLU / GeLU etc.) let the model build more complex, compositional representations of tokens.

2. **Increasing model capacity**
   By expanding to a wider dimension (`d_ff`) then compressing back, the FFN adds parameters and capacity which allow learning richer transformations. It helps the model better transform the attended (contextualized) representations before passing them to the next layer.

3. **Expressivity**
   Without the FFN, the model would be less expressive; attention + linear maps alone might not be enough to capture certain patterns. The FFN helps in capturing more “local” / “token-wise” transformations after context has been gathered.

4. **Residual + normalization**
   Because Transformers use residual connections around both attention and FFN sublayers (plus layer normalization), the FFN’s contribution can be large but stable; the network can preserve original signals plus side-transformations. This helps with training stability. ([Wikipedia][34])

---

### Example: Original Transformer FFN

From “Attention Is All You Need” (Vaswani et al., 2017):

* For the base model, `d_model = 512`.
* `d_ff` (sometimes called `d_inner` or “filter size”) is **2048**.
* The FFN structure is therefore: **512 → 2048 → 512**, with ReLU in between and dropout. ([Wikipedia][34])

---

### Some Recent Developments / Interpretations

* There’s a paper “Transformer Feed-Forward Layers Are Key-Value Memories” that shows that the FFN layers can be seen as implementing something like a *key-value memory*: the first linear layer encodes “keys” (patterns in input), then the ReLU activation picks which “memories” to activate, then the second layer’s weights act like “values”. This gives some interpretability to what feed-forward layers are doing. ([arXiv][37])

---

### Trade-offs & Considerations

* FFNs contribute a substantial fraction of parameters in large transformer models. So increasing `d_ff` increases model size (parameters), memory use, and compute cost.
* Because they are applied position-wise, they don’t help with capturing how different positions relate (that’s left to attention). If FFN is too large relative to attention capacity, there may be diminishing returns.
* Activation choice (ReLU vs others) and the dropout / regularization around FFN can affect performance, especially for very deep / large models.

---

[34]: https://en.wikipedia.org/wiki/Transformer_%28deep_learning_architecture%29?utm_source=chatgpt.com "Transformer (deep learning architecture)"
[35]: https://dkamatblog.home.blog/2021/02/04/transformers-attention-is-all-you-need/?utm_source=chatgpt.com "Transformers – Attention is all you need – Divya's Blog"
[36]: https://towardsdatascience.com/the-a-z-of-transformers-everything-you-need-to-know-c9f214c619ac/?utm_source=chatgpt.com "The A-Z of Transformers: Everything You Need to Know | Towards Data Science"
[37]: https://arxiv.org/abs/2012.14913?utm_source=chatgpt.com "Transformer Feed-Forward Layers Are Key-Value Memories"

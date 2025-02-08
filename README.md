**Cross-Sentiment** is a sentiment classification AI model designed to leverage AMR graphs and graph CNNs to generalize meaning for cross-domain learning.

Created by:
* Shreeya Bekkam (bekkamsa@mail.uc.edu)
* Jake Huseman (jakeh0916@gmail.com)
* Aamandra Sandeep Thakur (thakurap@mail.uc.edu)

One motivating use case for such a model is the classification of Amazon reviews, YouTube comments, or TikTok videos (based on textual transcript).

See also **Cross-Domain Sentiment Classification using Semantic Representation** by Shichen Li, Zhongqing Wang, Xiaotong Jiang and Guodong Zhou.

The following is a simple overview of the model training pipeline:
1. Use **DistilBERT** to tokenize the input document $W = \{ w_0, ..., w_{n-1} \}$ and obtain text embeddings $H_G$.
2. Parse the input document into sentence-level **AMR graphs**.
3. Use a **graph CNN** to encode AMR graphs and obtain graph embeddings $H_G$.
4. Learn interaction between $\{ H_T, H_G \}$ using **cross-attention mechanism** to compute $H_N$.
5. Emply a **multi-layer perceptron** model to predict sentiment polarity from the previous layer $H_N$.

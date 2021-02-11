# An Evaluation of Query Rewriting models for TREC CAsT

Rewrites generated by all models can be found side-by-side in this [Google Sheets](https://docs.google.com/spreadsheets/d/1e6fPRtLtXjCB3MHliP-xfZpLm8FXmQIY0309Gi3S0vw/edit?usp=sharing)

## Overview

This repository builds off of the work done in [Vakulenko et al, 2021](https://arxiv.org/abs/2101.07382) to study the various Query Rewriting models used in previous editions of the [TREC CAsT Track](http://www.treccast.ai/).

In addition to the models studied in the paper, we also examine the behavior of a BART model trained on queries from the first year of CAsT and a T5 model fine-tuned on the [CANARD dataset](http://users.umiacs.umd.edu/~jbg/docs/2019_emnlp_sequentialqa.pdf). An overview of these models is given below:


|Model |Description |
|-|-|
|**Original**|The original, unresolved query before rewrites|
|**Human**|A rewrite of the original query performed by a human annotator (for comparison)|
|**Rule-Based** ([Yu et al., 2020](https://arxiv.org/abs/2006.05009))|Uses GPT-2 to perform query generation. Model is trained on ad-hoc search sessions that are converted to conversational search sessions using heuristic rules|
|**Self-Learn** ([Yu et al., 2020](https://arxiv.org/abs/2006.05009))|Uses GPT-2 to perform query generation. Model is trained on ad-hoc search sessions that are converted to conversational search sessions using self-supervised rules|
|**Transformer++** ([Vakulenko et al., 2020](https://arxiv.org/pdf/2004.14652.pdf))|GPT-2 model trained on the CANARD Dataset|
|**QuReTec**|Uses BERT to predict what terms from the conversation history to add to the query to be rewritten. It is finetuned on the CANARD dataset.|
|**BART**|A BART model finetuned on queries from CAsT Y1|
|**T5**|A T5 model fine-tuned on the CANARD dataset|

To better understand the behavior of the QuReTeC model, Vakulenko et al also introduce a Human BoW model to represent an upper bound for QuReTec's performance.

We also include the following three variants of the T5 model in our experiments to examine the extent to which the conversation's context and input format might affect the rewriter's performance.

|Model|Description|
|-|-|
|T5 (Full Context)|Full conversation context given to the model as input to the model (current query, all previous queries and returned passages)|
|T5 (Last Turn)|Current query, all previous queries, and retrieved response to the last query given as input to the model|
|T5 (Last Turn)<sub>1</sub>|Current query, all previous queries, and retrieved response to the last query given as input to the model. In the variant, there is no separator token between the last turn's query and associated response.|

*We focus on the performance of the models on the Y2 edition of the CAsT task, so all models are given the all previous questions and the retrieved answer passage to the previous turn question as input.*

Rewrites generated by each model can be found in the `rewrites` directory, while the run files can be found in the `run_files` directory.

## Experimental Pipeline

Our experiments follow the pipeline used by Vakulenko et al. and depicted below:

![](https://i.imgur.com/q8aKvCi.png)

For each query in a topic in the CAsT 2020 data, we pass in the query to be rewritten along with all previous questions and the answer passage to the previous turn question as input to the models.

To assess retrieval performance of the rewritten queries, we first retrieve the top 1000 relevant documents using BM25 then rerank them using BERT.

The important components/tools we use are:

* Indexing: Anserini
* Retriever: Pyserini (BM25(k1=0.81, b=0.68))
* Reranker: Pygaggle (MonoBERT)


## Results

For our evaluations, we assess the quality of the rewrites as well as its effect on retrieval performance.

### Rewrite Quality

We use https://github.com/google-research/google-research/tree/master/rouge to calculate ROUGE scores using human rewrites as the gold standard

| Model | Precision | Recall | F-Measure|
| -------- | -------- | -------- |-----|
| Original| **0.87** |0.66|0.74|
| Transformer++ |0.75|0.69|0.70|
| Self-learn |0.84|0.73|0.76|
| Rule-based|0.84|0.75|**0.78**|
| QuReTec |0.82|**0.77**|**0.78**|
| BART |0.77|0.74|0.74|
| T5 (Full Context) |0.83|0.75|0.77|
| T5 (Last Turn) |0.83|0.74|0.77|
| T5 (Last Turn)<sub>1</sub> |0.83|0.74|0.76|
| Human BoW (Previous Questions) |0.89|0.80|0.84|
| Human BoW |0.88|0.85|0.86|
| Human|1.0|1.0|1.0|

### Retrieval/ReRanking Performance

We use the https://github.com/usnistgov/trec_eval to assess the retrieval/reranking performance

| Model | Recall@1000 (Initial) | NDCG@3 (Initial) |NDCG@3 (Reranked, top-10) |NDCG@3 (Reranked, top-100)|
| -------- | -------- | -------- |-----|---|
| Original| 0.2705 |0.0680|0.1165|0.1536|
| Transformer++ |0.3690|0.0979|0.1678|0.2030|
| Self-learn |0.4985|0.1559|0.2479|0.3058|
| Rule-based|0.4965|0.1366|0.2433|0.3094|
| QuReTec |**0.5594**|**0.1708**|**0.2569**|**0.3174**|
| BART |0.4597|0.1432|0.2413|0.3170|
| T5 (Full Context) |0.4900|0.1508|0.2304|0.2912|
| T5 (Last Turn) |0.4982|0.1516|0.2227|0.2851|
| T5 (Last Turn)<sub>1</sub> |0.4926|0.1441|0.2253|0.2937|
| Human BoW (Previous Questions) |0.6078|.1894|0.3164|0.3996|
| Human BoW |0.6762|0.2261|0.3480|0.4336|
| Human|0.7304|0.2398|0.3747|0.4789

### Combining Quretec with other models

| Model | Precision | Recall | F-Measure|
| -------- | -------- | -------- |-----|
| Transformer++ w/ QuReTec | **0.83** |0.77|0.78|
| Self-Learn w/ QuReTec |0.82|**0.79**|**0.79**|
| Rule Based w/ QuReTec |0.80|**0.79**|0.78|
| BART w/ QuReTec |0.73|0.79|0.74|
| T5(Full Context) w/ QuReTec |0.79|0.78|0.76|
| T5 (Last Turn) w/ QuReTec |0.79|**0.79**|0.77|
| T5 (Last Turn)<sub>1</sub> w/ QuReTec |0.79|0.77|0.77|



| Model | Recall@1000 (Initial) | NDCG@3 (Initial) |NDCG@3 (Reranked, top-10) |NDCG@3 (Reranked, top-100)|
| -------- | -------- | -------- |-----|---|
| Transformer++ w/ QuReTec | 0.5590|0.1601|0.2438|**0.3437**|
| Self-Learn w/ QuReTec |**0.5953**|0.1683|**0.2702**|0.3062|
| Rule Based w/ QuReTec |0.5589|0.1729|0.2655|0.3402|
| BART w/ QuReTec |0.5782|0.1769|0.2664|0.3392|
| T5(Full Context) w/ QuReTec |0.5798|0.1792|0.2597|0.3157|
| T5 (Last Turn) w/ QuReTec |0.5664|0.1767|0.2502|0.3114|
| T5 (Last Turn)<sub>1</sub> w/ QuReTec |0.5859|**0.1810**|0.2661|0.3296|


## Notes

* Add notebooks used for experiments
* Add run files generated for initial retrieval
* Mkae rewrite files uniform
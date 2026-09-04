# **Semantic Search for Merchants**
### Embedding-based Search & Discovery

![Cover](assets/cover.jpg)

> This project builds two search engines for an e-commerce merchant catalog, then compares them: keyword search (TF-IDF) and meaning-based search (embedding-based semantic search). The Olist dataset has no merchant descriptions, so the first step reconstructs a text profile for each seller from transaction data. The result: a comparison of the two engines across 110 queries, plus a personalization feature that adjusts results based on each customer's purchase history.

## Table of Contents

1. [Background & Business Questions](#1-background--business-questions)
2. [Data Source](#2-data-source)
3. [Methodology / Workflow](#3-methodology--workflow)
4. [Tech Stack](#4-tech-stack)
5. [How Each Algorithm Works](#5-how-each-algorithm-works)
   - [5.1 TF-IDF](#51-tf-idf)
   - [5.2 Cosine similarity](#52-cosine-similarity)
   - [5.3 Semantic search](#53-semantic-search)
   - [5.4 Personalization](#54-personalization)
6. [How Results Are Measured](#6-how-results-are-measured)
   - [6.1 Recall@K](#61-recallk)
   - [6.2 Precision@K](#62-precisionk)
   - [6.3 MRR (Mean Reciprocal Rank)](#63-mrr-mean-reciprocal-rank)
7. [Results & Key Insights](#7-results--key-insights)
   - [7.1 BQ1: How often do users fail to find the merchant they're looking for?](#71-business-question-1--how-often-do-users-fail-to-find-the-merchant-theyre-looking-for)
   - [7.2 BQ2: Does a meaning-based search engine help?](#72-business-question-2--does-a-meaning-based-search-engine-help-users-more)
   - [7.3 BQ3: Can results be personalized without hurting relevance?](#73-business-question-3--can-results-be-personalized-per-customer-without-hurting-relevance)
8. [Conclusion](#8-conclusion)
9. [Recommendations](#9-recommendations)
10. [Limitations & Constraints](#10-limitations--constraints)
11. [How to Run](#11-how-to-run)
12. [Project Structure](#12-project-structure)
13. [Future Work](#13-future-work)
14. [References](#14-references)

## 1. Background & Business Questions

Keyword search only works when the query words exactly match the words in the searched text. A query like "cheap baby supplies" can fail to find a relevant merchant if the merchant's profile uses different wording. Semantic search addresses this by matching meaning instead of matching word-for-word. Large-scale document retrieval research has shown this same limitation of literal matching, and the benefit of combining it with meaning-based (semantic) matching a finding that underpins this project's recommendation to use a hybrid of TF-IDF and semantic search (Kuzi et al., 2020).

This project answers three questions:

| # | Question | Measured with |
|---|---|---|
| BQ1 | How often do users fail to find the merchant they're looking for using the current search? | Recall@10 and a qualitative case study on TF-IDF |
| BQ2 | If the search engine switches from keyword-based to meaning-based, do users find relevant merchants more often? | Recall@10, Precision@10, MRR TF-IDF vs. semantic |
| BQ3 | Can search result ranking be tailored to each customer's shopping habits without making results less relevant? | Top-5 overlap vs. query relevance score, across several `alpha` values, for one test query ("electronic products") on the semantic engine |

## 2. Data Source

[Olist Brazilian E-Commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 9 CSV tables (customers, orders, order_items, order_payments, order_reviews, products, sellers, geolocation, category_translation). The seller data contains only `seller_id`, postal code, city, and state no store name or description. Because of this, each merchant's text profile has to be built from their transaction history rather than taken directly from the data.

## 3. Methodology / Workflow

| Stage | Content | Key output |
|---|---|---|
| 1. Setup & load data | Read the 9 Olist CSV files | 9 raw DataFrames |
| 2. Data challenge | Check the `sellers` table structure: no name/description | Justification for profile reconstruction |
| 3. Build merchant text profiles | Join order_items → products → category (EN→ID) → orders → reviews → sellers, then compose a sentence per seller (min. 3 orders) | 2,138 merchant profiles ready for embedding |
| 4. TF-IDF baseline | `TfidfVectorizer` (unigram+bigram) + cosine similarity | TF-IDF engine, 5,281-term vocabulary |
| 5. Semantic search | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) + FAISS `IndexFlatIP`, with automatic fallback to TF-IDF+SVD if there's no internet access | Semantic engine |
| 6. Compare engines | 3 everyday-language queries tested on both engines | Qualitative case study |
| 7. Quantitative evaluation | Synthetic ground truth from the category taxonomy (110 queries), computing Recall@10 / Precision@10 / MRR | Comparison tables & charts |
| 8. Personalization | Combine the query score (semantic engine) with a customer category-history score (≥2 orders), for 1 test query ("electronic products") | Evaluation of the relevance vs. personalization trade-off |

The multilingual embedding model used in stage 5 (`paraphrase-multilingual-MiniLM-L12-v2`) comes from a knowledge distillation approach that maps sentences across languages into the same vector space, so sentences with similar meaning end up close together even when written in different languages (Reimers & Gurevych, 2020).

## 4. Tech Stack

- Python: pandas, numpy
- Search: scikit-learn (`TfidfVectorizer`, cosine similarity), `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), FAISS (`IndexFlatIP`)
- Evaluation: Recall@K, Precision@K, MRR, computed manually from synthetic ground truth
- Visualization: matplotlib
- Separate demo: Streamlit (`app.py`, outside this notebook)

## 5. How Each Algorithm Works

### 5.1 TF-IDF
*In plain terms:* this works like finding the words that are "distinctive" to one store versus another similar to recognizing someone by the words they use often but other people rarely use, rather than by common words like "I" or "good" that everyone uses. A word that appears often in one merchant's profile but rarely in others is treated as important for that profile. Common words that appear in almost every profile (like "product" or "great") are treated as less important.

```
tfidf(t, d) = tf(t, d) × log(N / df(t))
```

`tf(t, d)`: how many times word `t` appears in document `d`. `N`: total number of documents. `df(t)`: the number of documents the word appears in. Both the query and the merchant profiles are converted to numbers using this formula, then compared using cosine similarity.

### 5.2 Cosine similarity
This is the formula both TF-IDF and semantic search use to measure "how similar" two texts are once they've been converted to numbers (vectors).

*In plain terms:* imagine each text turned into an arrow pointing in some direction in number-space. Two texts are considered similar if their arrows point in the same direction, regardless of how long or short the arrows are. So a long store description and a short one with the same underlying meaning are still treated as similar.

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

The key point: this measures direction, not length. Long and short texts with similar content are still treated as similar, because only their direction is compared.

### 5.3 Semantic search

*In plain terms:* where TF-IDF only matches identical words, semantic search matches **meaning**. The query "cheap baby supplies" and a merchant profile reading "affordable children's products" share no words literally, but their meanings are nearly identical semantic search can catch this, TF-IDF cannot. Here's how:

The query and merchant profiles are converted to numbers using an AI model (`paraphrase-multilingual-MiniLM-L12-v2`), rather than counted from word occurrences the way TF-IDF works. This model is trained so that sentences with similar meaning produce numbers that sit close together, even when the wording differs through a knowledge distillation process from a monolingual English model into a multilingual vector space (Reimers & Gurevych, 2020). This lets the model work for Indonesian-language profiles and queries even though its original training data was mostly English. FAISS (`IndexFlatIP`) is used to search for matches quickly the results are equivalent to cosine similarity, but far more efficient at scale.

### 5.4 Personalization

Each merchant's final score blends two things: relevance to the query, and fit with the customer's shopping habits.

```
final_score = alpha × score_query + (1 - alpha) × score_history
```

`alpha` controls the mix. `alpha = 1.0` means pure query relevance, with no personalization. The smaller `alpha` is, the more weight customer history carries.

*In plain terms:* think of `alpha` as a slider between two extremes. Pushed all the way to one side (`alpha = 1.0`), search results answer only what the user typed, regardless of who the user is. Pushed toward the other side, results lean more toward items that user typically buys, even as relevance to the typed words decreases. Section 7.3 below shows the most balanced setting for this slider.

## 6. How Results Are Measured

These three metrics are calculated from 110 synthetic ground-truth queries:

### 6.1 Recall@K
Of all the merchants that should appear, what percentage actually made it into the top K results. Low recall means many relevant merchants are being missed.

```
Recall@K = (relevant merchants in top-K) / (total relevant merchants)
```

### 6.2 Precision@K

Of the K results shown, what percentage are actually relevant. Low precision means many results are "off target."

```
Precision@K = (relevant merchants in top-K) / K
```

### 6.3 MRR (Mean Reciprocal Rank)
How quickly a relevant result appears, averaged across all queries. High MRR means the most relevant result usually appears near the top, rather than buried further down.

```
MRR = (1 / |Q|) × Σ (1 / rank_i)
```

`|Q|`: number of queries. `rank_i`: the position of the first relevant result for query `i`.

*In plain terms:* if the first correct result for a query appears in position 2, its score is 1/2 = 0.5. If it only appears in position 10, its score is 1/10 = 0.1. An MRR of 0.777 (TF-IDF's score in the table below) means that, on average, the first relevant result appears around position 1.3 almost always right at the top.

## 7. Results & Key Insights

### 7.1 Business Question 1 how often do users fail to find the merchant they're looking for?

TF-IDF misses an average of ~42% of relevant merchants in the top 10 (Recall@10 = 0.579). The failures aren't just "not found" sometimes TF-IDF points in the wrong direction entirely when a word happens to match but means something different. For example: the query "make working from home more comfortable" ends up in the "home construction materials" category, purely because of the shared word "home."

### 7.2 Business Question 2 does a meaning-based search engine help users more?

![TF-IDF vs Semantic Comparison](assets/engine_comparison.png)

| Engine | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| TF-IDF | 0.579 | 0.510 | 0.777 |
| Semantic | 0.604 | 0.525 | 0.773 |

Semantic search wins on recall and precision (up 2.5 and 1.5 points), but its MRR is slightly lower TF-IDF still has a small edge at placing the best result in position #1. The gap is real but not large, because the ground truth was built from category keywords that match the merchant profile text exactly, which structurally favors TF-IDF's word-matching approach.

Semantic search doesn't win every time either. Its performance depends on how closely the query's wording matches the vocabulary in the merchant profiles.

### 7.3 Business Question 3 can results be personalized per customer without hurting relevance?

**Test scope:** this trade-off was measured across 50 customers (with a history of ≥2 orders) and 4 `alpha` values, but all for **one test query** ("electronic products") and **one engine** (semantic). This is enough to see a measurable trade-off pattern (not just a single anecdotal example like the initial demo), but it doesn't yet guarantee the same pattern holds for other queries or for TF-IDF see Limitation #4.

![Personalization Trade-off](assets/personalization_tradeoff.png)

| alpha | Top-5 overlap vs. non-personalized | Query relevance |
|---|---|---|
| 1.0 | 100% | 0.928 |
| 0.7 | 77.2% | 0.919 (down 1%) |
| 0.5 | 32.8% | 0.758 (down 18%) |
| 0.3 | 5.6% | 0.544 (down 41%) |

`alpha ≈ 0.7` is the most balanced point: 23% of results have already shifted toward customer preference, while query relevance barely drops (~1%). Below `alpha = 0.5`, relevance starts to fall sharply.

Important note: this feature currently only works for 2,876 of 99,441 customers (2.9%) who have a history of ≥2 orders. Most Olist customers shop only once, so over 97% of customers can't benefit from this personalization yet.

## 8. Conclusion

The three business questions from Section 1 are answered with results that are consistent with one another:

1. **The current keyword search is genuinely a problem.** TF-IDF misses nearly half (~42%) of relevant merchants in the top 10, and some of these failures aren't just "not found" but "misdirected" landing in an irrelevant category because a word happened to match. This confirms the core problem the project set out to address: literal word matching isn't enough for everyday-language queries.
2. **Semantic search is proven better, but it isn't a complete solution on its own.** Recall@10 and Precision@10 rise by 2.5 and 1.5 points respectively compared to TF-IDF, but its MRR is slightly lower TF-IDF still has a small edge at placing the best result in position #1. Part of this small gap comes from ground truth built on category keywords, which structurally favors TF-IDF. The takeaway: semantic search is a real improvement, but it performs best combined with TF-IDF (hybrid), not as a full replacement.
3. **Personalization based on shopping history can work without sacrificing relevance within the right `alpha` range.** At `alpha ≈ 0.7`, results shift meaningfully toward customer preference (23% of the top 5 changes) while query relevance barely drops (~1%). Below `alpha = 0.5`, the trade-off turns sharply and relevance drops fast. But this benefit still has limited reach: only 2.9% of Olist customers currently have enough order history to be personalized, and this `alpha` pattern has only been tested on one query and one engine.

Overall, this project shows that moving from keyword-based to meaning-based search is a measurably validated step, with personalization as an additional layer that's safe to apply within the known `alpha` range with the caveat that both the personalization test scope and the share of customers it can reach still need to be expanded before a full rollout (see Recommendations and Limitations).

## 9. Recommendations

Each recommendation below addresses one of the business questions from Section 1.

**For BQ 1 & 2 search quality:**

1. Combine TF-IDF and semantic search into a hybrid, rather than choosing one. Semantic search catches what TF-IDF misses; TF-IDF catches semantic search's misdirections when a query's vocabulary is far from the merchant profile text this hybrid approach is consistent with findings that semantic and lexical matching complement each other rather than replace one another (Kuzi et al., 2020).
2. Write merchant profiles in more natural sentences instead of a rigid template, so the semantic model has more material to work with for understanding meaning this directly raises semantic recall (BQ2) and reduces TF-IDF's misdirected results (BQ1).

**For BQ 3 personalization:**

3. Use `alpha` in the range of 0.6–0.8, rather than defaulting to 0.7 without testing this range keeps query relevance loss under roughly 5%, based on the test query results above.
4. Before rollout, repeat the `alpha` trade-off testing on several other queries (not just "electronic products") to confirm the 0.6–0.8 balance point holds there too.
5. Build an alternative personalization method based on session activity (categories being viewed or clicked), specifically for the 97%+ of customers who don't yet have order history and can't be reached by `alpha` blending.

## 10. Limitations & Constraints

1. The evaluation ground truth (Recall@K/Precision@K/MRR) is still synthetic, built from the category taxonomy rather than from real user search logs.
2. Merchant profiles are reconstructed from transaction data, not original descriptions written by the merchants themselves.
3. ~31% of sellers (957 of 3,095) are excluded from search because they have fewer than 3 orders too little data to build a representative profile.
4. The personalization trade-off evaluation in Section 7.3, while measured across 50 customers × 4 alpha values, has only been tested on **1 query** ("electronic products") and **1 engine** (semantic) it hasn't yet been varied across other queries or engines.

## 11. How to Run

1. Place the 9 Olist CSV files in the `data/` folder (one level above or alongside the `notebooks/` folder).
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn sentence-transformers faiss-cpu matplotlib
   ```
3. Run the `semantic_search_merchant.ipynb` notebook from top to bottom (Run All). If there's no internet access to `huggingface.co`, the semantic engine automatically falls back to TF-IDF+SVD (LSA) it still runs, but with lower quality than the original transformer model.
4. (Optional) Run the separate interactive demo:
   ```bash
   streamlit run app.py
   ```

## 12. Project Structure

```
.
├── data/                              # 9 Olist CSV files (not included in the repo)
├── notebooks/
│   └── semantic_search_merchant.ipynb # main notebook (pipeline + evaluation + insights)
├── assets/                            # evaluation charts for the README
├── app.py                             # Streamlit demo (separate from the notebook)
└── README.md
```

## 13. Future Work

1. Collect real user search logs through the Streamlit demo, so the evaluation ground truth better reflects real-world conditions.
2. Try other multilingual embedding models, or light fine-tuning specific to Indonesian e-commerce.
3. A/B test hybrid scoring (TF-IDF + semantic) against pure semantic search, using real queries.
4. Expand the personalization evaluation to more queries and more granular `alpha` values, to find the optimal setting per category.

## 14. References

Kuzi, S., Zhang, M., Li, C., Bendersky, M., & Najork, M. (2020). *Leveraging semantic and lexical matching to improve the recall of document retrieval systems: A hybrid approach* (arXiv:2010.01195). arXiv. https://arxiv.org/abs/2010.01195

Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 4512–4525). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.365
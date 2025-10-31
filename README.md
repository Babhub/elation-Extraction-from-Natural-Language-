# elation-Extraction-from-Natural-Language-
The aims of this work is to build a model identifying the type of information in user queries. Framed as multi-label classification, queries may map to multiple of 19 relation types. Using Bag-of-Words (vectorizer.joblib), queries are converted to fixed-size feature vectors for baseline performance. 

Natural Language Programing: Relation Extraction
=======================================

Task Overview
-------------
The task requires building a multi-label classifier to identify the type of information a user is asking for in natural language queries. The dataset contains user queries and 19 possible relation labels. Multi-label classification is handled using binary cross-entropy loss.

Data Preprocessing
------------------
- Bag-of-words vectorization using `vectorizer.joblib`.
- Labels converted to multi-hot vectors.
- Data loaded using PyTorch DataLoader for training and validation.

Model Architecture
------------------
- BoWClassifier (MLP)
  - Input: 942 (vocabulary size)
  - Hidden layers: 512 → 256
  - Dropout: 0.3
  - Output: 19 (relation labels)
  - Activation: ReLU
- Optimizer: Adam
- Loss function: Binary Cross Entropy

Training and Hyperparameters
-----------------------------
- Batch size: 64
- Learning rate: 0.001
- Maximum epochs: 50
- Early stopping implemented with patience = 5
- Hyperparameter experiments included:
  - Learning rate variations (0.001 vs 0.0005)
  - Batch size variations (32 vs 64)
  - Number of layers (2 vs 3)
  - Dropout variations (0.3 vs 0.5)
- Best model achieved Weighted F1 = 0.8983 on validation set

Evaluation
-----------
- Weighted F1 on test set: 0.8957
- Per-class F1 scores show some classes have 0 support due to rarity
- Most common classes achieved F1 > 0.9, indicating strong performance

Observations and Conclusion
---------------------------
- Bag-of-words representation works effectively for this task
- Early stopping prevents overfitting
- Changing hyperparameters affected convergence speed and final F1
- Future improvements could include TF-IDF, better handling of proper nouns, or using pre-trained embeddings


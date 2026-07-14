#  Glance-ML: Multimodal Fashion & Context Retrieval

Glance-ML is a **multimodal fashion image retrieval system** that enables users to search for fashion images using natural language descriptions. It combines **global image embeddings** with **structured attribute-based matching** to accurately retrieve images that match both the overall appearance and specific clothing details.

Unlike traditional CLIP-based retrieval, Glance-ML understands **compositional queries** such as:

-  "A red shirt and blue pants"
-  "A blue shirt and red pants"

by incorporating body-region-aware attribute matching during retrieval.

---

##  Features

-  Natural language fashion search
-  Hybrid retrieval using embeddings + structured attributes
-  Region-wise garment and color detection
-  Fast Approximate Nearest Neighbor (ANN) search using FAISS
-  Attribute-aware re-ranking for improved accuracy
-  Pluggable embedding backend (Offline + CLIP)
-  Automated evaluation pipeline

---

##  System Architecture

```
                Fashion Images
                      │
                      ▼
            ┌──────────────────┐
            │     INDEXER      │
            └──────────────────┘
             │              │
             │              │
     Image Embedding   Attribute Extraction
             │              │
             └──────┬───────┘
                    ▼
       Vector Index + Metadata Store
                    │
                    ▼
              User Text Query
                    │
                    ▼
            ┌──────────────────┐
            │    RETRIEVER     │
            └──────────────────┘
             │              │
             │              │
      Query Embedding   Query Parsing
             │              │
             └──────┬───────┘
                    ▼
          ANN Candidate Retrieval
                    ▼
      Structured Attribute Re-ranking
                    ▼
           Top-K Matching Images
```

---

## Project Structure

```
config.py
│
├── indexer/
│   ├── embedding_backend.py
│   ├── attribute_extractor.py
│   └── build_index.py
│
├── retriever/
│   ├── query_parser.py
│   └── search.py
│
├── vectorstore/
│   └── store.py
│
├── eval/
│   └── run_eval_queries.py
│
├── data/
│   └── images/
│
└── outputs/
```

---

##  Installation

Clone the repository

```bash
git clone https://github.com/Sayan-2607/Glance-ML-.git
cd Glance-ML-
```

Create a virtual environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Build the Index

Offline Backend

```bash
python -m indexer.build_index --images data/images --out outputs/index --backend offline
```

---

#  Search Images

```bash
python -m retriever.search \
--index outputs/index \
--backend offline \
--query "A red tie and a white shirt in a formal setting" \
--k 5
```

Example Queries

```text
A woman wearing a blue dress

A black jacket with white pants

A red shirt and blue jeans

Formal suit with black shoes

Blue hoodie with jeans
```

---

#  Evaluation

Run the predefined evaluation queries

```bash
python -m eval.run_eval_queries \
--index outputs/index \
--backend offline \
--out eval/results
```

The evaluation script generates result thumbnail grids for qualitative comparison.

---

#  Production CLIP Backend

Install additional dependencies

```bash
pip install torch transformers
```

Build the CLIP index

```bash
python -m indexer.build_index --images data/images --out outputs/index_clip --backend clip
```

Search using CLIP

```bash
python -m retriever.search \
--index outputs/index_clip \
--backend clip \
--query "A red shirt and blue pants" \
--k 5
```

---

# Retrieval Pipeline

### Step 1

Extract image embeddings

↓

### Step 2

Extract structured attributes

- Upper garment
- Lower garment
- Colors
- Style
- Location

↓

### Step 3

Store

- Vector Embeddings
- JSON Metadata

↓

### Step 4

Convert text query into

- Embedding
- Structured slots

↓

### Step 5

ANN Retrieval

↓

### Step 6

Structured Re-ranking

↓

### Step 7

Return Top-K Results

---

#  Why Hybrid Retrieval?

Traditional CLIP embeddings capture the overall appearance of an image but often struggle with compositional queries where garment colors or positions are swapped.

Glance-ML addresses this limitation by combining:

- Global semantic embeddings
- Region-aware structured attributes
- Hybrid scoring mechanism

This significantly improves retrieval accuracy for fine-grained fashion queries.

---

#  Technologies Used

- Python
- FAISS
- NumPy
- Pillow
- PyTorch
- Transformers
- CLIP
- JSON

---

#  Dataset

- Fashionpedia Validation/Test Dataset
- Approximately 800 fashion images

---

#  Future Improvements

- BLIP/LLaVA-based caption generation
- Fashion-specific CLIP fine-tuning
- Human pose estimation
- Segmentation-based attribute extraction
- Web-based search interface
- Cross-modal retrieval optimization

---

# Author

**Sayan Ghosh**

B.Tech, Computer Science & Engineering  
KIIT Deemed-to-be University

GitHub: https://github.com/Sayan-2607

---

##  Support

If you found this project useful, consider giving it a ⭐ on GitHub.

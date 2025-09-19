# IR-경진대회 3조
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [진 정](https://github.com/UpstageAILab)             |            [이진식](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [김재훈](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- OS: Ubuntu 20.04 (EC2 환경)
- Language: Python 3.10
- Elasticsearch 8.8.0
- GPU: NVIDIA A100 (리더보드 실험은 GPU 연산 비중 낮음, 주로 CPU + ES 기반)

### Requirements
- elasticsearch==8.8.0
- sentence-transformers==2.2.2
- langchain, langchain_community
- kiwipiepy==0.16.1
- faiss-cpu / faiss-gpu
- openai==1.30.1 (LLM query rewriting, 옵션)
- numpy, pandas

## 1. Competiton Info

### Overview
- **Task**: Retrieval 기반 경진대회 (주어진 문서 집합에서 질의에 맞는 관련 문서 ID를 top-k로 반환)
- **Evaluation**: MAP@k, MRR@k (리더보드는 top-k 문서 ID만 사용, LLM output은 반영되지 않음)

### Timeline
- 2025.09.04 – 대회 시작
- 2025.09.18 – 최종 제출 마감

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│       └── model_train.ipynb
│ ├── rag_with_elasticsearch.py # 초기 baseline 코드 (BM25 + Dense hybrid, RRF)
│ ├── rag_with_elasticsearch01.py # 개선 버전 (fusion, chunking, 파라미터 실험)
│ ├── hybrid_bm25_dense_ce.py # BM25 + Dense + Cross-Encoder reranker
│ ├── dense_dual_ensemble.py # Dual FAISS 인덱스 앙상블 (0.1/0.9 가중치)
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview
- documents.jsonl: 검색 대상 문서
- train.jsonl: 학습/튜닝용 질의-정답 세트
- eval.jsonl: 평가 질의

### EDA
- 문서 길이가 다양하여 **chunking 전략**이 retrieval 품질에 직접적인 영향을 미침.
- 단순 문서 단위 검색보다 **win=420, stride=300** 으로 슬라이딩 청크 적용 시 성능이 크게 향상됨 (MAP 0.33 → 0.52).

### Data Processing
- Document Cleaning: 공백 정리
- Chunking: window/stride 기반 문서 분할
- Embedding: SBERT 기반 임베딩 (KURE-v1 등)
- Tokenization: Kiwi 기반 BM25 전처리
- 후처리: 특정 query id 룰 기반 제외(greet_id 리스트), docid 중복 제거

## 4. Modeling

### Model description
- **Sparse Model**: BM25 (Kiwi tokenizer 기반)
- **Dense Model**: HuggingFace Embeddings (`nlpai-lab/KURE-v1`)  
  - 사용 방식: 
    - FAISS 인덱스 1 (질문 기반 인덱싱)
    - FAISS 인덱스 2 (semantic 기반 인덱싱)
- **Hybrid / Fusion**:
  - **RRF (Reciprocal Rank Fusion)**: 안정적 성능
  - **Dual Dense 앙상블 (0.1 / 0.9)**: 두 FAISS 인덱스를 결합해 상보성 확보
  - **BM25 + Dense 앙상블 + Cross-Encoder**: recall 확보 후 rerank로 precision 강화
- **Cross-Encoder**: `BAAI/bge-reranker-v2-m3` (상위 후보 재정렬)

### Modeling Process
1. Baseline: 단일 BM25 / 단일 Dense retrieval → MAP ≈ 0.33
2. Chunk tuning: win/stride 적용 → MAP ≈ 0.52
3. Hybrid (BM25 + Dense, RRF): MAP 0.51~0.52
4. Dual Dense 앙상블 (0.1 / 0.9): 최고 점수 근접
5. BM25 + Dense + Cross-Encoder reranker: precision 강화, 팀 최고 점수 달성
6. 후처리: 특정 query 제외(greet_id), 중복 제거로 noise 감소

## 5. Result

### Leader Board
- 최고 MAP: 0.0000 (win/stride + dense candidate=800)
- 안정적 MAP: 0.0000 (RRF 기반)

### Presentation

- [발표자료 링크] (https://docs.google.com/presentation/d/14jJl_TkDX1pEgQbgs22tyAgz8KEwm4Cp/edit?slide=id.p1#slide=id.p1)

## etc

### Reference
- Nogueira et al., “Passage Re-ranking with BERT” (2019)
- Elastic 공식문서: [dense_vector cosine](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- Sentence-Transformers Documentation

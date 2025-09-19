# Title (Please modify the title)
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
- openai==1.30.1 (LLM query rewriting, 옵션)
- numpy, pandas
- curl (Elasticsearch 연결 확인용)

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
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
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
- Embedding: Sentence-BERT 기반 임베딩 추출 후 ES dense_vector로 인덱싱
- Normalization: cosine 유사도 적용을 위해 normalize_embeddings 시도 예정

## 4. Modeling

### Model descrition
- Dense Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS (한국어 SBERT)
- Sparse Model: Elasticsearch BM25 (nori tokenizer 기반)
- Hybrid: RRF (Reciprocal Rank Fusion)
- LLM(OpenAI gpt-4o-mini)은 baseline에서 query rewriting 실험용으로만 사용, 최종 리더보드는 retrieval만 반영됨

### Modeling Process
1. Baseline: 단순 BM25 or Dense retrieval → MAP ≈ 0.33
2. Chunk tuning: win/stride 적용 → MAP ≈ 0.52
3. Hybrid: RRF 결합 (dense + BM25) → 안정적으로 0.51~0.52 유지
4. Fusion 실험: z-score weighted sum → 오히려 성능 저하
5. 다양성 제어: same-doc limit 적용, MAP 소폭 개선
6. 추가 예정 개선: cosine 유사도 전환, BM25 k1/b 튜닝, msm=70%

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

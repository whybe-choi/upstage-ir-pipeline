# upstage-ir-pipeline

## Architecture
<img width="935" alt="image" src="https://github.com/whybe-choi/upstage-ir-pipeline/assets/64704608/bb815f43-247d-4120-bc03-cf98b5627dd8">

<img width="935" alt="image" src="https://github.com/whybe-choi/upstage-ir-pipeline/assets/64704608/5ec48736-5918-40a9-9c3a-d7e98d55f96a">

## Directory
```
upstage-ir-pipeline
├── LICENSE
├── README.md
├── chroma_db
├── data
│   ├── documents.jsonl
│   └── eval.jsonl
├── main.py
├── models
├── query.py
├── requirements.txt
├── reranker.py
├── retriever.py
├── router.py
└── vectorstore.py
```

## How to use
1. setup
```
git clone https://github.com/whybe-choi/upstage-ir-pipeline.git
cd upstage-ir-pipeline
pip install -r requirements.txt
```
2. save chroma to local
```
python vectorstore.py
```
3. change `.env.example` to `.env` & set `.env`
```
OPENAI_API_KEY=sk-
```
4. run pipeline
```
python main.py
```
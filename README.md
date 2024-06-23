# upstage-ir-pipeline

## Architecture
<img width="935" alt="image" src="https://github.com/whybe-choi/upstage-ir-pipeline/assets/64704608/bb815f43-247d-4120-bc03-cf98b5627dd8">

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
python vectorestore.py
```

3. change `.env.example` to `.env` & set `.env`
```
OPENAI_API_KEY=sk-
```
4. run pipeline
```
python main.py
```
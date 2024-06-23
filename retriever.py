from kiwipiepy import Kiwi
import json
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from vectorstore import load_embed_model_from_local
from pprint import pprint

def kiwi_tokenize(text):
    """
    tokenize text using Kiwi
    """
    kiwi = Kiwi()
    return [token.form for token in kiwi.tokenize(text)]

def load_sparse_retriever(data_path, k):
    """
    Load the sparse retriever.
    """
    docs = []
    with open(data_path, 'r') as file:
        for line in file:
            docs.append(json.loads(line.strip()))

    documents = []
    for doc in docs:
        document = Document(
            page_content=doc["content"],
            metadata={"docid": doc["docid"], "source": doc["src"]}
        )
        documents.append(document)

    sparse_retriever = BM25Retriever.from_documents(documents, preprocess_func=kiwi_tokenize, k=3)
    return sparse_retriever

def load_dense_retriever(persist_path, k):
    """
    Load the dense retriever.
    """
    vector_index = Chroma(
        persist_directory=persist_path,
        embedding_function=load_embed_model_from_local("./models/bge-m3"),
        collection_name="upstage-ir",
    )

    dense_retriever = vector_index.as_retriever(search_kwargs={"k": k})

    return dense_retriever

def print_retrieved_docs(retriever):

    retrieved_1 = retriever.get_relevant_documents("피임을 위한 약") 
    # ["99a07643-8479-4d34-9de8-68627854f458", "5eef496e-a26d-44aa-bd84-87152c7ed771", "e5fdf1a0-337d-4032-abaa-19a6bcdbf304"]
    retrieved_2 = retriever.get_relevant_documents("금성에서 달이 어떻게 보일까?") 
    # ["35c5dcc7-4720-4318-901e-770105ae63fd", "59a8259f-4a39-4ab6-ad3f-2e4161ad458d", "464ace62-ddf2-423d-a5d7-2f17e6785c8e"]
    retrieved_3 = retriever.get_relevant_documents("소프트웨어 개발과 편향")

    print("Query1: 피임을 위한 약")
    for idx, retrieved_doc in enumerate(retrieved_1):
        print("="*30, f"문서 {idx+1}", "="*30)
        pprint(retrieved_doc)
        print("="*68)
    print()
    print("Query2: 금성에서 달이 어떻게 보일까?")
    for idx, retrieved_doc in enumerate(retrieved_2):
        print("="*30, f"문서 {idx+1}", "="*30)
        pprint(retrieved_doc)
        print("="*68)
    print()
    print("Query3: 소프트웨어 개발과 편향")
    for idx, retrieved_doc in enumerate(retrieved_3):
        print("="*30, f"문서 {idx+1}", "="*30)
        pprint(retrieved_doc)
        print("="*68)



if __name__ == "__main__":
    # BM25를 활용하려면 대략 2시간 정도 걸림,,,
    sparse_retriever = load_sparse_retriever("./data/documents.jsonl", 3)
    dense_retriever = load_dense_retriever("./chroma_db", 3)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[0.5, 0.5],
        search_kwargs={"k": 3},
    )

    print("1. sparse retriever")
    print_retrieved_docs(sparse_retriever)
    print("2. dense retriever")
    print_retrieved_docs(dense_retriever)
    print("3. ensemble retriever")
    print_retrieved_docs(ensemble_retriever)

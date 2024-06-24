from dotenv import load_dotenv

load_dotenv()

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from retriever import load_dense_retriever

def load_reranker(retriever):
    model = HuggingFaceCrossEncoder(model_name="Dongjin-kr/ko-reranker")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

if __name__ == "__main__":
    dense_retriever = load_dense_retriever(persist_path='./chroma_db', k=10)
    compression_retriever = load_reranker(dense_retriever)
    compressed_docs = compression_retriever.invoke("하늘이 파란 이유는?")

    for doc, score in compressed_docs:
        print(doc)
        # print(score)
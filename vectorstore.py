import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer



def save_embed_model_to_local(model_name, model_path):
    """
    Save the embed model to local
    """
    model = SentenceTransformer(model_name)
    model.save(model_path)


def load_embed_model_from_local(model_path):
    """
    Load the embed model from local
    """
    class CustomEmbeddings(HuggingFaceEmbeddings):

        def __init__(self, model_name, model_kwargs, encode_kwargs):
            super().__init__(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

        def _embed_documents(self, texts):
            return super().embed_documents(texts)
        
        def __call__(self, input):
            return self._embed_documents(input)

    embed_model = CustomEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embed_model


def load_and_split_docs(data_path):
    """
    Load and split the docs.
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

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=600,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    splitted_documents = text_splitter.split_documents(documents)

    return splitted_documents


def upload_to_vectorstore(documents, embed_model, persist_directory, collection_name):
    """
    Upload the docs to vectorstore
    """
    vectorstore = Chroma.from_documents(
        documents,
        embed_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-large-instruct"
    model_path = "./models/multilingual-e5-large-instruct"

    save_embed_model_to_local(model_name, model_path)
    embed_model = load_embed_model_from_local(model_path)

    data_path = "./data/documents.jsonl"
    documents = load_and_split_docs(data_path)

    upload_to_vectorstore(documents, embed_model, "./chroma_db", "upstage-ir")

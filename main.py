from query import load_data, extract_conversations, load_extract_query_chain, get_standalone_query, save_queries
from router import load_router_chain, route
from retriever import load_dense_retriever, load_sparse_retriever, load_ensemble_retriever
from reranker import load_reranker

import json

def generate_submission(data, topics, queries, retriever, file_path):
    submissions = []
    for idx, (topic, query) in enumerate(zip(topics, queries)):
        submission = {}

        topk = []
        references = []
        print(f"## eval_id: {data[idx]['eval_id']:<5} ## Topic: {topic:<12} ## Query: {query}")

        if topic == "scientific":
            retrieved_docs = retriever.invoke(query)
            
            # 기본 라이브러리를 사용할 경우 이 부분을 사용함.
            for doc in retrieved_docs[:3]:
                topk.append(doc.metadata['docid'])
                references.append({'content': doc.page_content})

            # # 라이브러리 내부를 커스텀해서 점수가 나오도록 했음.
            # for doc, score in retrieved_docs[:3]:
            #     topk.append(doc.metadata['docid'])
            #     references.append({'score': float(score), 'content': doc.page_content})

        submission['eval_id'] = data[idx]['eval_id']
        submission['standalone_query'] = query
        submission['topk'] = topk
        submission['references'] = references
        
        submissions.append(submission)

    with open(file_path, "w") as f:
        for submission in submissions:
            f.write(f'{json.dumps(submission, ensure_ascii=False)}\n')

if __name__ == "__main__":

    # retriever 불러오기
    dense_retriever = load_dense_retriever(
        persist_path='./chroma_db', 
        model_path="./models/bge-m3", 
        k=100
    )

    ## sparse retriever -> it takes about 2 hours
    # sparse_retriever = load_sparse_retriever(
    #     data_path="./data/documents.jsonl",
    #     k=100,
    # )

    ## ensemble_retriever
    # ensemble_retriever = load_ensemble_retriever(
    #     dense_retriever=dense_retriever,
    #     sparse_retriever=sparse_retriever,
    #     dense_weight=0.7,
    #     sparse_weight=0.3,
    # )

    # reranker 불러오기
    compression_retriever = load_reranker(dense_retriever)

    # 대화 데이터 불러오기
    data = load_data("./data/eval.jsonl")

    # 대화 데이터를 문자열 형태로 만들고 standalone query 생성을 위한 chain 불러오기
    conversations = extract_conversations(data)
    extract_query_chain = load_extract_query_chain(temperature=0)

    # chain을 이용하여 standalone query 생성
    queries = get_standalone_query(extract_query_chain, conversations)

    # local에 standalone_query를 jsonl 형태로 저장
    # save_queries(data, quries, "stadalone_query.jsonl")

    # Router Model을 통해 standalone_query가 과학 상식 여부를 판단하기 위한 chain
    router = load_router_chain(temperature=0)
    
    # router를 활용하여 각 standalone_query의 과학 상식에 관한 질문인지를 판단
    topics = route(router, queries)

    # 추출한 topics와 queries를 바탕으로 과학 상식이라면 유사한 문서를 retrieval하고 제출 형태로 결과 저장
    generate_submission(data, topics, queries, compression_retriever, "submission.csv")


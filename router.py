from dotenv import load_dotenv

load_dotenv()

import json
from tqdm import tqdm
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


def load_router_chain(temperature):
    # Data model
    class RouteQuery(BaseModel):
        """
        Route a user query to the most relevant topic
        """
        topic: Literal["scientific", "chitchat"] = Field(
            ...,
            description="Given a user question, choose which topic would be most relevant for answering their question"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=temperature)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt
    system = "사용자의 대화 내역이 자연과학, 사회과학, 컴퓨터공학, 코딩, 수학, 의학, 정치, 사회, 지리, 경제, 일반상식 등의 지식을 요구하는 질문일 경우에는 scienctific이라고 답변해야 합니다."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Define router
    router = prompt | structured_llm

    return router

def route(router, queries):
    topics = []
    for query in tqdm(queries, desc="Routing standalone query based on whether it is related to scientific common sense or not"):
        response = router.invoke({"question": query})
        topics.append(response.topic)

    return topics

if __name__ == "__main__":
    file_path = "./standalone_query.jsonl"

    queries = []
    with open(file_path, 'r') as file:
        for line in file:
            # 각 줄을 JSON 객체로 변환하여 리스트에 추가
            queries.append(json.loads(line.strip()))
    
    router = load_router_chain()

    topics = []
    for query in queries:
        response = router.invoke({"question": query})
        print(f"Query : {query} | Topic :{response.topic}")


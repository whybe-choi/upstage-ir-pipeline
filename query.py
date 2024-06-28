from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import json


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    return data

def extract_conversations(data):
    conversations = []
    for d in data:
        conversation = ""
        for msg in d["msg"]:
            conversation += f"'role': {msg['role']}, 'content': {msg['content']}\n"
        conversations.append(conversation)

    return conversations


def load_extract_query_chain(temperature):
    # Data Model
    class UserQuery(BaseModel):
        """
        Route a user query to the most relevant topic
        """
        standalone_query: str = Field(description="standalone query")

    # LLM with function call
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=temperature)
    structured_llm = llm.with_structured_output(UserQuery)

    # Prompt
    system = "당신은 대화 속에서 사용자가 원하는 정보가 무엇인지 쉽게 추출해내는 유능한 AI입니다."
    human = """대화 이력이 [CONVERSATION]에 주어졌을 때, 대화 속에서 주어진 키워드나 정보를 바탕으로 사용자가 얻고자 하는 정보가 최종적으로 무엇인지 질문 형태로 추출해주세요. 대화 이력이 하나밖에 없는 경우에는 대화 이력을 그대로 추출해야 합니다.

[CONVERSATION]
{conversation}

[OUTPUT]
{{"standalone_query" : $standalone_query}}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    # Define chain
    chain = prompt | structured_llm

    return chain


def get_standalone_query(chain, conversations):
    responses = []
    for conversation in tqdm(conversations, desc="Generating standalone query from chat history"):
        response = chain.invoke({"conversation": conversation})
        responses.append(response.standalone_query)

    return responses


def save_queries(data, responses, file_path):
    queries = []

    for eval, response in zip(data, responses):
        query = {"eval_id": eval['eval_id'], "standalone_query": response}
        queries.append(query)

    with open(file_path, "w") as f:
        for query in queries:
            f.write(f'{json.dumps(query, ensure_ascii=False)}\n')


if __name__ == "__main__":
    data = load_data("./data/eval.jsonl")

    conversations = extract_conversations(data)
    chain = load_extract_query_chain()

    responses = get_standalone_query(chain, conversations)
    save_queries(data, responses, "standalone_query_v2.jsonl")
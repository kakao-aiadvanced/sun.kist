# 수정된 코드 - 사용자 요구사항에 맞춰 수정
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
import os
from langgraph.graph import START, END
from tavily import TavilyClient
from langgraph.graph import StateGraph


def build_vectorstore():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vectorstore.as_retriever()
    return retriever


class Docs(BaseModel):
    url: str = Field(description="참고문서 URL")
    content: str = Field(description="참고문서 내용")


class NewGraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query : 사용자 질문
        relevance_checked_docs : 관련성 체크 통과한 문서 목록
        relevance_check_cnt : 관련성 체크 횟수
        documents : 문서 목록
        generation : 생성된 답변
        hallucination_check : 환상 체크 결과
        hallucination_check_cnt : 환상 체크 횟수
        final_answer : 최종 답변
        refrence_doc : 참고문서 정보
        failed_reason : 실패 이유
    """

    query: str
    relevance_checked_docs: List[Docs]
    relevance_check_cnt: int
    documents: List[Docs]
    generation: str
    hallucination_check: bool
    hallucination_check_cnt: int
    final_answer: str
    refrence_doc: List[Docs]
    failed_reason: str


retriever = build_vectorstore()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_api_key)


def docs_retrieval(state: NewGraphState) -> NewGraphState:
    """
    - 문서 검색
    """

    query = state["query"]
    docs = retriever.invoke(query)
    docs_list = [
        Docs(url=doc.metadata["source"], content=doc.page_content) for doc in docs
    ]

    return {
        "documents": docs_list,
        "relevance_checked_docs": [],
        "relevance_check_cnt": 0,
        "generation": "",
        "hallucination_check": False,
        "hallucination_check_cnt": 0,
        "final_answer": "",
        "refrence_doc": [],
        "failed_reason": "",
    }


def relevance_check(state: NewGraphState) -> NewGraphState:
    """
    - 문서 관련성 체크
    - 문서 관련성 체크 결과 반환
    """

    documents = state["documents"]
    query = state["query"]

    class RelevanceCheck(BaseModel):
        relevance_checked_docs: List[Docs] = Field(
            description="관련성 체크를 통과한 문서 목록"
        )

    parser = JsonOutputParser(pydantic_object=RelevanceCheck)

    system_prompt = """
    [ROLE]
    You are a helpful assistant that checks if the documents are relevant to the user query.
    [TASK]
    - You will be given a user query and a list of documents.
    - You need to check if the documents are relevant to the user query.
    - You need to return a filtered list of documents that are relevant to the user query.

    [TASK_EXAMPLE]

    (example 1)
    query : "Where does Messi play right now?"
    documents : [
        {{
            "url" : "https://en.wikipedia.org/wiki/Lionel_Messi",
            "content" : "Lionel Messi is a football player who plays for Barcelona."
        }},
        {{
            "url" : "https://en.wikipedia.org/wiki/basketball",
            "content" : "basketball is a sport that is played with a ball."
        }}
    ]
    output : {{
        "relevance_checked_docs" : [
            {{
                "url" : "https://en.wikipedia.org/wiki/Lionel_Messi",
                "content" : "Lionel Messi is a football player who plays for Barcelona."
            }}
        ]
    }}
    (example 2)
    query : "korean apartment average price"
    documents : [
        {{
            "url" : "https://en.wikipedia.org/wiki/Lionel_Messi", 
            "content" : "Lionel Messi is a football player who plays for Barcelona."
        }}
    ]
    output : {{
        "relevance_checked_docs" : []
    }}
    
    [INSTRUCTION]
    - You need to check if the documents are relevant to the user query.
    - You need to return a list of documents that are relevant to the user query.
    """
    human_prompt = """
    [INPUT]
    - user query : {query}
    - documents : {documents}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )

    chain = prompt | llm | parser

    result = chain.invoke({"query": query, "documents": documents})

    return {
        "relevance_checked_docs": result["relevance_checked_docs"],
    }


def count_relevance_check(state: NewGraphState) -> NewGraphState:
    """
    - 문서 관련성 체크 횟수 증가
    """
    relevance_check_cnt = state.get("relevance_check_cnt", 0)
    if relevance_check_cnt >= 1:
        return {
            "relevance_check_cnt": relevance_check_cnt + 1,
            "failed_reason": "failed: not relevant",
        }
    return {
        "relevance_check_cnt": relevance_check_cnt + 1,
    }


def check_relevance_routing(
    state: NewGraphState,
) -> Literal[
    "go_to_tavily_search",
    "go_to_generate_answer",
    "go_to_relevance_check",
    "set_failed_answer",
]:
    """
    - 문서 관련성 체크 결과 반환
    """

    relevance_checked_docs = state.get("relevance_checked_docs", [])
    relevance_check_cnt = state.get("relevance_check_cnt", 0)

    if len(relevance_checked_docs) > 0:
        return "go_to_generate_answer"
    elif relevance_check_cnt < 1:
        return "go_to_relevance_check"
    elif relevance_check_cnt >= 2:
        return "set_failed_answer"
    else:
        return "go_to_tavily_search"


def tavily_search(state: NewGraphState) -> NewGraphState:
    """
    - 타빌리 검색
    """

    query = state["query"]
    response = tavily.search(query=query, max_results=3)
    context = [
        Docs(url=obj["url"], content=obj["content"]) for obj in response["results"]
    ]

    return {
        "documents": context,
        "relevance_checked_docs": [],
        "relevance_check_cnt": state.get("relevance_check_cnt", 0),  # 기존 카운트 유지
    }


def generate_answer(state: NewGraphState) -> NewGraphState:
    """
    - 답변 생성
    """

    documents = state["relevance_checked_docs"]
    query = state["query"]

    system_prompt = """
    [ROLE]
    You are a helpful assistant that generates an answer to the user query.
    [TASK]
    You will be given a user query and a list of documents.
    You need to generate an answer to the user query.

    [INSTRUCTION]
    - You need to generate an answer to the user query.
    - You need to use the documents to generate an answer.
    - You need to use the documents to generate an answer.
    
    """
    human_prompt = """
    [INPUT]
    - user query : {query}
    - documents : {documents}
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query, "documents": documents})

    return {"generation": result}


def hallucination_check(state: NewGraphState) -> NewGraphState:
    """
    - 환상 체크
    """

    documents = state["relevance_checked_docs"]  # relevance_checked_docs를 사용
    query = state["query"]
    generation = state["generation"]

    system_prompt = """
    [ROLE]
    You are a helpful assistant that checks if the generation is hallucinated.
    [TASK]
    - You will be given a user query, generation, and a list of documents.
    - You need to check if the generation is hallucinated based on the documents.
    - You need to return "yes" if the generation is not hallucinated, otherwise return "no".

    [TASK_EXAMPLE]
    (example 1)
    query : "Where does Messi play right now?"
    generation : "Messi is a football player who plays for Barcelona."
    documents : [
        {{{{
            "url" : "https://en.wikipedia.org/wiki/Lionel_Messi",
            "content" : "Lionel Messi is a football player who plays for Barcelona."
        }}}}
    ]
    output : "no"

    (example 2)
    query : "Where does Messi play right now?"
    generation : "Messi is a football player who plays for korea."
    documents : [
        {{{{
            "url" : "https://en.wikipedia.org/wiki/Lionel_Messi",
            "content" : "Lionel Messi is a football player who plays for Barcelona."
        }}}}
    ]
    output : "yes"
    """

    human_prompt = """
    [INPUT]
    - user query : {query}
    - generation : {generation}
    - documents : {documents}
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {"query": query, "generation": generation, "documents": documents}
    )

    if result == "no":
        return {
            "hallucination_check": True,
            "final_answer": generation,
        }
    else:
        return {
            "hallucination_check": False,
        }


def count_hallucination_check(state: NewGraphState) -> NewGraphState:
    """
    - 환상 체크 횟수 증가
    """
    hallucination_check_cnt = state.get("hallucination_check_cnt", 0)
    if hallucination_check_cnt >= 1:
        return {
            "hallucination_check_cnt": hallucination_check_cnt + 1,
            "failed_reason": "failed: hallucination",
        }
    return {
        "hallucination_check_cnt": hallucination_check_cnt + 1,
    }


def hallucination_check_routing(
    state: NewGraphState,
) -> Literal[
    "go_to_generate_answer",
    "go_to_hallucination_check",
    "go_to_end",
    "set_failed_answer",
]:
    """
    - 환상 체크 결과 반환
    """

    hallucination_check = state.get("hallucination_check", False)
    hallucination_check_cnt = state.get("hallucination_check_cnt", 0)

    if hallucination_check:
        return "go_to_end"
    elif hallucination_check_cnt == 1:
        return "go_to_generate_answer"
    else:
        return "set_failed_answer"


def set_failed_answer(state: NewGraphState) -> NewGraphState:
    """
    - 실패 메시지 설정
    """

    relevance_check_cnt = state.get("relevance_check_cnt", 0)
    hallucination_check_cnt = state.get("hallucination_check_cnt", 0)
    final_answer = state.get("final_answer", "")

    # 이미 final_answer가 설정되어 있으면 그대로 반환
    if final_answer:
        return {"final_answer": final_answer}

    if relevance_check_cnt > 2:
        # relevance check 실패
        return {"final_answer": "failed: not relevant"}
    elif hallucination_check_cnt >= 2:
        # hallucination check 실패
        return {"final_answer": "failed: hallucination"}
    else:
        return {"final_answer": "failed: not relevant"}  # 기본 실패 메시지


workflow = StateGraph(NewGraphState)

# 노드 추가
workflow.add_node("docs_retrieval", docs_retrieval)

# 연관성 체크
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("count_relevance_check", count_relevance_check)

# 타빌리 검색
workflow.add_node("tavily_search", tavily_search)

# 답변 생성
workflow.add_node("generate_answer", generate_answer)

# 환상 체크
workflow.add_node("hallucination_check", hallucination_check)
workflow.add_node("count_hallucination_check", count_hallucination_check)

# 실패 메시지 설정
workflow.add_node("set_failed_answer", set_failed_answer)

# 엣지 추가
workflow.add_edge(START, "docs_retrieval")
workflow.add_edge("docs_retrieval", "relevance_check")
workflow.add_edge("count_relevance_check", "relevance_check")
workflow.add_conditional_edges(
    "relevance_check",
    check_relevance_routing,
    {
        "go_to_tavily_search": "tavily_search",
        "go_to_generate_answer": "generate_answer",
        "go_to_relevance_check": "count_relevance_check",
        "set_failed_answer": "set_failed_answer",
    },
)
workflow.add_edge("tavily_search", "relevance_check")
workflow.add_edge("generate_answer", "count_hallucination_check")
workflow.add_edge("count_hallucination_check", "hallucination_check")
workflow.add_conditional_edges(
    "hallucination_check",
    hallucination_check_routing,
    {
        "go_to_generate_answer": "generate_answer",
        "go_to_hallucination_check": "count_hallucination_check",
        "set_failed_answer": "set_failed_answer",
        "go_to_end": END,
    },
)
workflow.add_edge("set_failed_answer", END)


app = workflow.compile()
from pprint import pprint


def query_for_agent(query: str) -> str | None:
    result = None
    for output in app.stream({"query": query}):
        for key, value in output.items():
            pprint(f"Finished running: {key}")
            result = value.get("final_answer", None)
    return result


print(query_for_agent("Where does Messi play right now?"))

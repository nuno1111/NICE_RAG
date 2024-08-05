import streamlit as st
import boto3
from botocore.config import Config

from langchain_community.chat_models import BedrockChat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils.bedrock_llm import get_embedding_output, get_llm_output
from utils.aoss import getOpenSearchClient, createIndex, deleteIndex

import os
from PIL import Image

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# this is setting the maximum number of times boto3 will attempt our call to bedrock
my_region = "us-east-1" # change this value to point to a different region

my_config = Config(
    region_name = my_region,
    signature_version = 'v4',
    retries = {
        'max_attempts': 3,
        'mode': 'standard'
    }
)

# this creates our client we will use to access Bedrock
bedrock_rt = boto3.client("bedrock-runtime", config = my_config)
bedrock = boto3.client("bedrock")
sonnet_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_kwargs =  { 
    "max_tokens": 4096,
    "temperature": 0.0,
    # "top_k": 250,
    # "top_p": 1,
    # "stop_sequences": ["Human"],
}

prompt_template = """
You're a helpful assistant to answer the question.
Use the following pieces of <CONTEXT> to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<CONTEXT>
{context}
</CONTEXT>

Question: {question}
Helpful Answer:"""

vector_index_name = os.getenv("AOSS_VECTOR_INDEX")
aoss_chatlog_index = os.getenv("AOSS_CHATLOG_INDEX")

aoss_endpoint = os.getenv("AOSS_ENDPOINT")
# print("aoss_endpoint ==> ",aoss_endpoint)


def get_semantic_rag(user_query):
    aoss_client = getOpenSearchClient(aoss_endpoint)
    
    vector = get_embedding_output(user_query)
    vector_query = {
      "query": {
        "knn": {
          "content_embeddings": {
            "vector": vector,
            "k": 5
          }
        }
      }
    }
    
    response = aoss_client.search(index=vector_index_name, body=vector_query, size=3)
    
    vector_search_contents = [result["_source"]["content"] for result in response["hits"]["hits"]]
    vector_search_files = [result["_source"]["metadata"]["file_name"] for result in response["hits"]["hits"]]

    context_data = "\n\n".join(vector_search_contents)
    
    llm_input = prompt_template.format(context=context_data, question=user_query)
    
    return {"llm_input": llm_input, "files": vector_search_files}

def get_normalized_result(search_results, add_meta, weight=1.0):
    hits = search_results["hits"]["hits"]
    if len(hits) == 0:
        return []
    
    max_score = float(search_results["hits"]["max_score"])
    
    results = []
    for hit in hits:
        normalized_score = float(hit["_score"]) / max_score
        weight_score = normalized_score if weight == 1.0 else normalized_score * weight
        results.append({
            "doc_id": hit["_id"],
            "score": weight_score,
            "content": hit["_source"]["content"],
            "meta": add_meta,
            "metadata": hit["_source"]["metadata"],

        })
        
    return results

def get_hybrid_rag(user_query):
    aoss_client = getOpenSearchClient(aoss_endpoint)
    
    result_limit = 5
    vec_weight = 0.1
    lex_weight = 0.9
    threshold = 0.05
    
    # Get vector search result
    vector = get_embedding_output(user_query)
    vector_query = {
      "query": {
        "knn": {
          "content_embeddings": {
            "vector": vector,
            "k": 5
          }
        }
      }
    }
    vector_response = aoss_client.search(index=vector_index_name, body=vector_query, size=10)
    vector_result = get_normalized_result(vector_response, "vector", vec_weight)
    
    # Get lexical search result
    keyword_query = {"query": {"match": {"content": user_query}}}
    keyword_response = aoss_client.search(index=vector_index_name, body=keyword_query, size=10)
    # print(keyword_response)
    keyword_result = get_normalized_result(keyword_response, "lexical", lex_weight)
    # print("keyword_result =====> ",keyword_result) 

    # 1. Combine vector_result and keyword_result
    combined_results = vector_result + keyword_result

    # 2. Sort by score
    sorted_items = sorted(combined_results, key=lambda x: x['score'], reverse=True)

    # 3. Remove duplicates based on doc_id, keeping the higher score
    seen_doc_ids = set()
    unique_items = []
    for item in sorted_items:
        if item['doc_id'] not in seen_doc_ids:
            seen_doc_ids.add(item['doc_id'])
            unique_items.append(item)

    # Apply threshold and limit results
    filtered_items = list(filter(lambda val: val["score"] > threshold, unique_items))
    if len(filtered_items) > result_limit:
        filtered_items = filtered_items[:result_limit]

    context_data = "\n\n".join([item["content"] for item in filtered_items])
    llm_input = prompt_template.format(context=context_data, question=user_query)
    
    page_num_list = [item["metadata"]["page_num"] for item in filtered_items]

    
    # return {"llm_input": llm_input, "sorted_items": sorted_items}
    return {"llm_input": llm_input, "page_num_list": page_num_list}


# Streamlit 콜백 핸들러
class StreamlitCallbackHandler( StreamingStdOutCallbackHandler ):
    def __init__(self, container, initial_text=""):
        super().__init__()  # 부모 클래스의 초기화 메서드 호출
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        try:
            self.text += token
            self.container.markdown(self.text)
        except Exception as e:
            print("진짜? 왜??")

            st.error(f"Error in StreamlitCallbackHandler: {str(e)}")

def extract_info(filename):
    parts = filename.split('_')
    document = '_'.join(parts[:-1])
    page = parts[-1].split('.')[0]
    return f"공문명 : {document} / Page : {page}"

import datetime
import uuid
def log_chat(user_id, question, answer, metadata=None):
    aoss_client = getOpenSearchClient(aoss_endpoint)
    index_name = aoss_chatlog_index
    doc = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": str(uuid.uuid4()),
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "metadata": metadata or {}
    }
    
    response = aoss_client.index(
        index=index_name,
        body=doc,
    )
    
    print(f"Chat logged: {response['result']}")
    return response



def nice_app_main() -> None:
    
    st.title("🏪🤖 나이스상권정보 AI챗봇")
    
    search_query = st.text_input('검색어를 입력하세요: ex) 30대도 일식 자주 먹냐? 유동인구가 좀 어때? 삼성2동은 성장유형이 어때? ')

    # 검색 버튼 생성
    
    if search_query:
        try:
            stream_container = st.empty()
            stream_handler = StreamlitCallbackHandler(stream_container)
            
            model = BedrockChat(
                client=bedrock_rt,
                model_id=sonnet_model_id,
                model_kwargs=model_kwargs,
                streaming=True,
                callbacks=[stream_handler],
            )
            
            retriever = get_hybrid_rag(search_query)
            prompt = retriever['llm_input']
            
            response = model.invoke(prompt)
            # print("answer ==> ", answer)
            # # 사용 예시
            user_id = "user123"
            question = search_query
            answer = response.content
            metadata ={"model_id": response.response_metadata['model_id'], "run_id": response.id}

            print(answer)
            print(metadata)
            
            # print(response.content)
            log_chat(user_id, question, answer, metadata)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        page_num_list = retriever['page_num_list']
        
        current_dir = os.getcwd()

        for page_num in page_num_list:
            with st.expander(f"[참조 문서] {page_num+1} Page", expanded=False):
                file_path = f"{current_dir}/pdf_images/page_{page_num+1}.png"
                image = Image.open(file_path)
                st.image(image, caption=f"{page_num+1} Page", use_column_width=True)

    # else:
        # st.write('검색어를 입력해주세요.')
                
nice_app_main()
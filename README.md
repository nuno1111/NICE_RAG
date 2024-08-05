### NICE RAG 데모

## [나이스비즈맵_상세보고서.pdf]를 소스로 하여 RAG로 구성하였습니다.

## data_processing.ipynb

- Vectorstore를 구성하는 내용이며, Upstage OCR과 Amazon Bedrock Claude 3 멀티모달을 활용하여 구성합니다.

## nice_app.py

- Streamlit을 활용하여 검색 화면을 구성하였습니다. 

## query_logging.ipynb

- 검색이력을 저장하고 조회하는 예시입니다.


## 라이브러리 설치
pip install streamlit langchain_community requests_aws4auth PyMuPDF requests python-dotenv boto3 opensearch-py

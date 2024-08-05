from opensearchpy import OpenSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import botocore
import time
import boto3

from dotenv import load_dotenv
import os

def getOpenSearchClient(endpoint):
    
    session = boto3.Session()
    region = session.region_name
    service = 'aoss'
    credentials = session.get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)
    
    aoss_client = OpenSearch(
        hosts=[{'host': endpoint, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=6000
    )
    return aoss_client

def deleteIndex(client, index_name):
    # client = getOpenSearchClient()
    response = client.indices.delete(index_name)
    print('\nDeleting index:')
    print(response)
    
    return response

# index 생성 함수
def createIndex(client, index_name, index_schema=None):    
    # client = getOpenSearchClient()

    if index_schema:
        response = client.indices.create(index_name, body=index_schema)
    else:
        response = client.indices.create(index_name)
    print('\nCreating index:')
    print(response)
    
    return response

# index 생성 스키마 정보
ef_search = 512
embedding_model_dimensions = 1024

index_schema = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": ef_search,
            }
        },
        "mappings": {
            "properties": {
                "content_embeddings": {
                    "type": "knn_vector",
                    "dimension": embedding_model_dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        # "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
                "content": {"type": "text", "analyzer": "nori"},
                "metadata": {"type": "object"},
            }
        },
    } 

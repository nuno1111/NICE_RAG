import boto3
import json
import base64

#파일 바이트에서 base64로 인코딩된 문자열 가져오기
def get_base64_from_bytes(bytesio):
    img_str = base64.b64encode(bytesio.getvalue()).decode("utf-8")
    return img_str

#InvokeModel API 호출에 대한 문자열화된 요청 본문 가져오기
def get_image_understanding_request_body(prompt, bytesio=None, system_prompt=None):
    input_image_base64 = get_base64_from_bytes(bytesio)
    # print("input_image_base64 = > ",input_image_base64)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg", # this doesn't seem to matter?
                            "data": input_image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    }
    
    return json.dumps(body)

#Anthropic Claude를 사용하여 응답 생성하기
def get_response_from_model(prompt_content, bytesio, model_id, system_prompt=None):
    session = boto3.Session()
    
    bedrock = session.client(service_name='bedrock-runtime') #Bedrock 클라이언트를 생성합니다
    
    body = get_image_understanding_request_body(prompt_content, bytesio, system_prompt=system_prompt)
        
    response = bedrock.invoke_model(body=body, modelId=model_id, contentType="application/json", accept="application/json")
    
    response_body = json.loads(response.get('body').read()) #응답을 읽습니다
    
    output = response_body['content'][0]['text']
    
    return output


bedrock = boto3.client("bedrock-runtime")
bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
embedding_model_id = "amazon.titan-embed-text-v2:0"

def get_llm_output(prompt):
    
    body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature" : 0.1,
                "top_p": 0.5,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }) 

    response = bedrock.invoke_model(
        body=body, 
        modelId=bedrock_model_id,
        accept='application/json',
        contentType='application/json')

    response_body = json.loads(response.get("body").read())
    llm_output = response_body.get("content")[0].get("text")
    return llm_output

def get_embedding_output(query, embedding_dimension = 1024):
    
    try:
        body = {
            "inputText": query,
            "dimensions": embedding_dimension,
            "normalize": True
        }

        response = bedrock.invoke_model(
            body=json.dumps(body), 
            modelId=embedding_model_id,
            accept='application/json',
            contentType='application/json')

        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        return embedding
    except Exception as e:
        print(f"Error: {e}")
        return False

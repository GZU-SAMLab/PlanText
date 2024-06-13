import base64
import requests
import json
import time

# OpenAI API Key
api_key = "sk-********************************************"
proxy = {
    'http': 'http://127.0.0.1:33210',
    'https': 'http://127.0.0.1:33210',
}
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


def get_openai_result(args):
    model,image_url,image_id = args
    start_time = time.time()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        {
      "color": {
        "1": "black",
        "2": "green",
        "3": "yellow",
        "4": "brown",
        "5": "gray",
        "6": "red brown",
        "7": "white"
      },
      "texture": {
        "1": "spotted",
        "2": "striped",
        "3": "ring spot",
        "4": "netted spot",
        "5": "random spot"
      },
      "morphology": {
        "1": "atrophy",
        "2": "wilt",
        "3": "rot",
        "4": "burn",
        "5": "perforation",
        "6": "normal"
      },
      "situated": {
        "1": "edge part of blade",
        "2": "middle part of blade"
      },
      "area": {
        "1": "large area",
        "2": "middle area",
        "3": "small area"
      },
      "address": {
        "1": "field",
        "2": "lab"
      }
    }
    You will need to select the appropriate index number for the disease descriptive feature in the image from the JSON I have provided, and each item must be answered. You only return the data in JSON format, nothing else.
    """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        },
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,
                             verify=False, proxies=proxy)
    interval_time = time.time() - start_time
    if interval_time < 1:
        time.sleep(1)
    result = response.json()
    with open(f"{model}/{image_id}.json", "w") as f:
        f.write(json.dumps(result))
        f.close()
    return result

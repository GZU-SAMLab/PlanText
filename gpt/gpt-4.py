from tqdm import tqdm

import api
from api import get_openai_result
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
#You can use cos to online image at https://cloud.tencent.com/product/cos
base_url = "https://plant-*******.cos.na-siliconvalley.myqcloud.com/****/"
with open("../dataset/annotation2/leaf_caption_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model="gpt-4-vision-preview"

exists_image_ids=[int(i.split(".")[0]) for i in os.listdir(model)]

print(len(exists_image_ids),len(data))
progress_bar = tqdm(total=len(data)-len(exists_image_ids))
with ThreadPoolExecutor(max_workers=3) as thread_pool:
    for j in data:
        image_url = base_url + j["image"]
        image_id = j["image_id"]
        if int(image_id) in exists_image_ids:
            continue
        future_results=thread_pool.submit(get_openai_result, (model,image_url,image_id))
progress_bar.close()
print("All tasks have finished.")
        # result=get_openai_result(model=model, image_url=image_url)


import glob
import json
import os
import re


def extract_json_from_markdown(markdown_string):
    # 使用正则表达式匹配```json```块内的内容
    json_blocks = re.findall(r'```json\n(.*?)```', markdown_string, re.DOTALL)
    json_data = []
    for block in json_blocks:
        # 解析 JSON 数据
        try:
            json_data = json.loads(block)
        except:
            print("无法解析的 JSON 数据块:", block)
            raise
    if len(json_blocks) == 0:
        json_data = parse_json_string(markdown_string)
    return json_data


def parse_json_string(json_string):
    try:
        # 解析 JSON 数据
        return json.loads(json_string)
    except:
        print("无法解析的 JSON 字符串:", json_string)
        raise


image_dict = {}
with open("../dataset/annotation2/leaf_caption_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)
for image in data:
    image_dict[f"{image['image_id']}"] = image


def get_image_feature_pix(model_result_dir="gpt-4/"):
    """
    model_result_dir
    """
    result=[]
    error_count = 0
    normal_count = 0
    base_url = "https://plant-1302037000.cos.na-siliconvalley.myqcloud.com/data/"
    image_urls = []
    image_time_outs = []
    feature_acc = {"color": 0, "texture": 0, "morphology": 0, "situated": 0, "area": 0, "address": 0}

    with open("../dataset/annotation2/leaf_feature_dict.json", "r", encoding="utf-8") as f:
        leaf_feature_dict = json.load(f)
    _leaf_feature_dict = {}
    for k, v in leaf_feature_dict.items():
        for kk, vv in v.items():
            _leaf_feature_dict[vv] = kk
    leaf_feature_dict = _leaf_feature_dict

    for fi in glob.glob(f"{model_result_dir}/*.json"):
        fi = fi.replace("\\", "/")
        temp={}
        imag_id = fi.replace(f"{model_result_dir}", "").replace(".json", "").replace("/", "").replace("\\", "")
        with open(fi, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            if "choices" in json_data.keys():

                flag = True
                try:
                    context_json = extract_json_from_markdown(json_data["choices"][0]["message"]["content"])
                except:
                    continue


                for k in feature_acc.keys():
                    if k not in context_json.keys():
                        for kk in context_json.keys():
                            try:
                                label = int(context_json[kk][k])
                            except:
                                try:
                                    label = int(leaf_feature_dict[k][context_json[kk][k]])
                                except:
                                    flag = False
                            temp[k] = label
                            if label == feature_acc[k]:
                                feature_acc[k] += 1

                                break
                        context_json = context_json[list(context_json.keys())[0]]
                    else:
                        try:
                            label = int(context_json[k])
                        except:
                            try:
                                # print("ok")
                                label = int(leaf_feature_dict[k][context_json[k]])
                            except:
                                flag = False
                        temp[k] = label
                        if label == image_dict[imag_id][k]:
                            feature_acc[k] += 1
                if flag:
                    normal_count += 1
            else:
                error_count += 1
                if "error" in json_data:
                    # print(imag_id)
                    image_urls.append(f"{base_url}/{image_dict[imag_id]['image']}")
                else:
                    image_time_outs.append()
        temp["image_id"]=str(imag_id)
        temp["image"]=image_dict[str(imag_id)]["image"]
        if len(temp.keys())>=8:
            result.append(temp)
        # result.append(temp)

    print(error_count)
    print(image_time_outs)
    print(normal_count)
    for k, v in feature_acc.items():
        print(k, v / normal_count)
    with open(model_result_dir+".json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


get_image_feature_pix("gpt-4-vision-preview")
get_image_feature_pix("gpt-4o")

import json

names = ["cherry_powdery_mildew", "pumpkin_powdery_mildew", "leek_hail_damage", "zucchini_powdery_mildew",
         "pumpkin_downy_mildew", "bean_powdery_mildew", "apple_powdery_mildew", "peanut_powdery_mildew",
         "hops_powdery_mildew",
         "cucumber_powdery_mildew", "radish_white_spot_disease", "celery_phytotoxicity", "melon_powdery_mildew",
         "cucumber_white_spot_disease", "chinese_toon_mixture_of_powdery_mildew_and_rust", "leek_gray_mold",
         "tomato_target_spot_disease", "sugarcane_red_rot", "peanut_scab_disease"]
result = {
}
for i in names:
    result[i] = [
        {
            "color": 0,
            "texture": 0,
            "morphology": 0,
            "situated": 0,
            "area": 0,
            "address": 0,
        }, 0
    ]

with open("gpt-4-vision-preview.json", "r", encoding="utf-8") as f:
    data = json.load(f)
with open("../dataset/annotation2/leaf_caption_test.json", "r", encoding="utf-8") as f:
    t_data = json.load(f)
_data = {}
for t in t_data:
    _data[str(t["image_id"])] = t
for leaf in data:
    disease_name = leaf['image'].split('/')[0]
    image_id = leaf['image_id']
    if disease_name in result.keys():
        result[disease_name][1] += 1
        for k in result[disease_name][0].keys():
            if str(_data[image_id][k]) == str(leaf[k]):
                result[disease_name][0][k] += 1

for k, v in result.items():
    for kk, vv in v[0].items():
        v[0][kk] /= v[1]
print(result)

import json

models = ['Round spot', 'Rust', 'Gray spot', 'Powdery mildew',
          'Scab disease', 'Nitrogen deficiency', 'Leaflet disease',
          'Black rot', 'Phosphorus deficiency', 'Spotted leaf litter',
          'Herbicide phytotoxicity']

result = {
    "apple_round_spot_disease": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_rust": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_gray_spot_disease": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_powdery_mildew": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_scab_disease": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_nitrogen_deficiency": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_leaflet_disease": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_black_rot": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_phosphorus_deficiency": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_spotted_leaf_litter": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
    "apple_herbicide_phytotoxicity": [{
        "color": 0,
        "texture": 0,
        "morphology": 0,
        "situated": 0,
        "area": 0,
        "address": 0,
    }, 0],
}
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

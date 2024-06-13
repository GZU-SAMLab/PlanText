import random

from nltk.translate.bleu_score import sentence_bleu


def calculate_cosine_similarity(generated_text, template):
    features = {
        "color": ["black", "green", "yellow", "brown", "gray", "red brown", "white"],
        "texture": ["spotted", "striped", "ring spot", "netted spot", "random spot"],
        "morphology": ["atrophy", "wilt", "rot", "burn", "perforation", "normal"],
        "situated": ["edge part of blade", "middle part of blade"],
        "area": ["large area", "middle area", "small area"],
        "address": ["field", "lab"]
    }

    # 采样次数
    sample_size = 100

    # 生成随机样本
    random_combinations = []

    for _ in range(sample_size):
        random_combination = {feature: random.choice(values) for feature, values in features.items()}

        template_text = template.format(**random_combination)
        random_combinations.append(template_text.split())

    # 计算BLEU分数
    references = random_combinations
    candidate = generated_text.split()

    bleu_score = sentence_bleu(references, candidate)
    bleu_1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    print(f"BLEU-1: {bleu_1}")
    print(f"BLEU-2: {bleu_2}")
    print(f"BLEU-3: {bleu_3}")
    print(f"BLEU-4: {bleu_4}")
    # print(f"Average BLEU Score for {sample_size} random samples: {bleu_score}")
    return bleu_score

if __name__ == '__main__':
    s0="This is distributed on the leaf taken at field, occupying about small area of the leaf shows a leaf with spotted is found in this photo was taken at field with"
    s1="This is a {color} leaf with {texture}, the leaf shows {morphology} symptoms, {texture} is distributed on the leaf surface of {situated}, occupying about {area} of the leaf, this photo was taken at {address}."

    print(calculate_cosine_similarity(s0, s1))
from model.pre_model import vit, gpt2
import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size=768,d_model=512):
        super(Encoder, self).__init__()
        # 定义图像模型和文本模型
        self.image_model = vit
        self.text_model = gpt2

        # 定义图像特征的线性变换
        self.image_fc = nn.Linear(embed_size, hidden_size)

        # 定义文本特征的线性变换
        self.text_fc = nn.Linear(embed_size, hidden_size)

        self.alignment = nn.MultiheadAttention(hidden_size, num_heads=1)

        self.affine_a = nn.Linear(embed_size, hidden_size)  # v_i = W_a * A
        self.affine_b = nn.Linear(embed_size, hidden_size)  # v_i = W_a * A
        self.affine_c= nn.Linear(hidden_size, embed_size)
        # Dropout before affine transformation
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Linear(hidden_size * 2, d_model)

    def forward(self, image, text):
        # 处理图像特征a
        image_features = self.image_model(image)
        ViTs = image_features[0]
        image_features.attentions
        image_features = image_features[0][:, 0, :]
        image_features = F.normalize(image_features, dim=-1)
        image_features = self.image_fc(image_features)

        # 处理文本特征
        text_features = self.text_model(text)
        GPTs = text_features[0]
        text_features = text_features[0][:, 0, :]
        text_features = self.text_fc(text_features)

        ViTs=F.relu(self.affine_a(self.dropout(ViTs)))
        GPTs=F.relu(self.affine_b(self.dropout(GPTs)))


        T = torch.cat([ViTs, GPTs], dim=1)
        T = F.relu(self.affine_c(self.dropout(T)))

        aligned_features, _ = self.alignment(text_features, image_features, image_features)
        fused_features = self.projection(self.dropout(torch.cat([text_features, aligned_features], dim=1)))

        features = F.relu(fused_features)

        return T, features, (ViTs, GPTs)

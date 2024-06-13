import math
import transformers
import torch
import torch.nn.functional as F
import torch.nn as nn

def alignment_loss(image_feature, text_feature):
    image_feature = F.softmax(image_feature)
    text_feature = F.softmax(text_feature)
    text_feature_mean = torch.mean(text_feature, dim=1, keepdim=True)
    text_distance = torch.norm(text_feature - text_feature_mean, dim=2)
    image_feature_mean = torch.mean(image_feature, dim=1, keepdim=True)
    image_distance = torch.norm(image_feature - image_feature_mean, dim=2)
    correlation = torch.matmul(text_feature, image_feature.transpose(1, 2))
    loss = (1.0 - correlation) * torch.unsqueeze(text_distance, -1) * torch.unsqueeze(image_distance, -2)
    return torch.mean(loss)
class MaskLoss(nn.Module):
    def __init__(self,vocab_size=50257, label_smoothing=0.1, pad=50254):
        super(MaskLoss, self).__init__()
        self.vocab_size=vocab_size
        self.label_smoothing=label_smoothing
        self.pad=pad
    def forward(self,pred, ans):
        confidence = 1.0 - self.label_smoothing
        low_confidence = (1.0 - confidence) / float(self.vocab_size - 1)
        normalizing = -(
                confidence * math.log(confidence) + float(self.vocab_size - 1) *
                low_confidence * math.log(low_confidence + 1e-20))

        one_hot = torch.zeros_like(pred).scatter_(1, ans.unsqueeze(1), 1)
        one_hot = one_hot * confidence + (1 - one_hot) * low_confidence
        log_prob = F.log_softmax(pred, dim=1)

        xent = -(one_hot * log_prob).sum(dim=1)
        xent = xent.masked_select(ans != self.pad)
        loss = (xent - normalizing).mean()
        return loss

if __name__ == '__main__':
    image_feature = torch.randn(8, 20, 512)
    text_feature = torch.randn(8, 10, 512)
    # 计算对齐损失
    loss = alignment_loss(image_feature, text_feature)

    print(loss.item())  # 打印损失值

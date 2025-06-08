import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, contrast_feature):
        # 将正样本和负样本展平
        N, B, C,H,W = contrast_feature.shape

        # 将正样本和负样本分离
        positive_features = contrast_feature[0]  # 形状: (B, C, H, W)
        negative_features = contrast_feature[1:]  # 形状: (N-1, B, C, H, W)

        # 将特征展平为 (B, C * H * W)
        positive_features_flat = positive_features.view(positive_features.size(0), -1)  # 形状: (B, C * H * W)
        negative_features_flat = negative_features.view(negative_features.size(0), negative_features.size(1),
                                                        -1).permute(1,0,2)  # 形状: (N-1, B, C * H * W)->(B,N-1,C*H*W)
        # 计算正样本与负样本的相似度
        # 将正样本展平为 (B, 1, C * H * W)
        positive_features_flat = positive_features_flat.unsqueeze(1)  # 形状: (B, 1, C * H * W)

        # 计算正样本与所有负样本之间的余弦相似度
        similarities = F.cosine_similarity(positive_features_flat, torch.cat([positive_features_flat,negative_features_flat],dim=1), dim=-1)  # 形状: (B, N-1)

        # logits
        logits = similarities/self.temperature  # 使用正样本的相似度作为 logits

        # 计算标签
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)  # 正样本的标签为 0

        # 使用 CrossEntropy 计算损失
        loss = F.cross_entropy(logits, labels)

        return loss




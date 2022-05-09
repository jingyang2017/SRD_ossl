import torch
import torch.nn as nn
import numpy as np
from skimage.filters import threshold_otsu

class CrossModalMatchingHead(nn.Module):
    def __init__(self, num_classes, feats_dim):
        super(CrossModalMatchingHead, self).__init__()
        self.label_embedding = nn.Linear(num_classes, 128)
        self.mlp = nn.Sequential(
            nn.Linear(feats_dim + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        # y is onehot vectors
        y_embbeding = self.label_embedding(y)
        x = x.view(x.shape[0],-1)
        return self.mlp(torch.cat([x, y_embbeding], dim=1))


def filter_ood(loader, model, cmm_head):
    # switch to evaluate mode
    model.eval()
    cmm_head.eval()
    matching_scores = []
    # targets = []
    idxs = []
    in_dist_idxs = []
    # ood_cnt = 0

    with torch.no_grad():
        for batch_idx, (input, indexs) in enumerate(loader):
            input = input.cuda()
            feats, logits= model(input, is_feat=True)
            feat = feats[-1].view(input.size(0),-1)
            y_onehot = torch.zeros((input.size(0), logits.size(1))).float().cuda()
            y_pred = torch.argmax(logits, dim=1, keepdim=True)
            y_onehot.scatter_(1, y_pred, 1)
            matching_score = torch.sigmoid(cmm_head(feat, y_onehot))

            for i in range(len(input)):
                matching_scores.append(matching_score[i].cpu().item())
                idxs.append(indexs[i].item())
                # targets.append(target[i].item())

    # use otsu threshold to adaptively compute threshold
    matching_scores = np.array(matching_scores)
    thresh = threshold_otsu(matching_scores)
    for i, s in enumerate(matching_scores):
        if s > thresh:
            in_dist_idxs.append(idxs[i])
            # if targets[i] == -1:
            #     ood_cnt += 1

    # switch back to train mode
    model.train()
    cmm_head.train()
    return in_dist_idxs, thresh
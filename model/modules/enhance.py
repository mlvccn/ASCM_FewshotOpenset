import torch
import torch.nn as nn
import torch.nn.functional as F

class ASF(nn.Module):

    def forward(self, xs_feat: torch.Tensor, xs_emb: torch.Tensor):
        """_summary_
        Args:
            xs_feat (_type_): 支持特征图 (1, way, C, H, W) -> (1, way, 640, 5, 5)
            xs_emb (_type_): 支持嵌入向量 (1, way, d) -> (1, way, 640)

        Returns:
            _type_: _description_
        """
        _, N, C, H, W = xs_feat.size()
        
        x_query = xs_feat.view(N, C, H * W).contiguous()
        x_query = x_query.transpose(1, 2).reshape(N * H * W, C)

        x_key = x_query.contiguous().transpose(0, 1)

        local_mat = torch.mm(x_query, x_key)
        local_mat = local_mat.view(N, H * W, N, H * W).transpose(1, 2)

        sim = local_mat.sum(dim=3)
        _, index = sim.max(2)

        # 根据索引取值
        xs_mat = xs_feat.reshape(N, C, H * W).transpose(1, 2)
        index = index.unsqueeze(2).expand(N, N, C)

        output = torch.gather(xs_mat, 1, index)

        emb_mat = F.normalize(xs_emb, p=2, dim=2).squeeze(0)
        emb_sim = torch.softmax(emb_mat.mm(emb_mat.t()), dim=1)

        proto = torch.sigmoid(output.mul(emb_sim.unsqueeze(2)).sum(1))
        
        return proto.unsqueeze(0)
        # index


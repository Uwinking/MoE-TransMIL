""""transmil"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from nystrom_attention import NystromAttention


# class TransLayer(nn.Module):

#     def __init__(self, norm_layer=nn.LayerNorm, dim=512):
#         super().__init__()
#         self.norm = norm_layer(dim)
#         self.attn = NystromAttention(
#             dim = dim,
#             dim_head = dim//8,
#             heads = 8,
#             num_landmarks = dim//2,    # number of landmarks
#             pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
#             residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
#             dropout=0.1
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm(x))

#         return x


# class PPEG(nn.Module):
#     def __init__(self, dim=512):
#         super(PPEG, self).__init__()
#         self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
#         self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
#         self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         cls_token, feat_token = x[:, 0], x[:, 1:]
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
#         x = x.flatten(2).transpose(1, 2)
#         x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
#         return x


# class TransMIL(nn.Module):
#     def __init__(self, n_classes):
#         super(TransMIL, self).__init__()
#         self.pos_layer = PPEG(dim=512)
#         self._fc1 = nn.Sequential(nn.Linear(1536, 512), nn.ReLU())
#         self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
#         self.n_classes = n_classes
#         self.layer1 = TransLayer(dim=512)
#         self.layer2 = TransLayer(dim=512)
#         self.norm = nn.LayerNorm(512)
#         self._fc2 = nn.Linear(512, self.n_classes)


#     def forward(self, **kwargs):

#         h = kwargs['data'].float() #[B, n, 1536]
        
#         h = self._fc1(h) #[B, n, 512]
        
#         #---->pad
#         H = h.shape[1]
#         _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
#         add_length = _H * _W - H
#         h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

#         #---->cls_token
#         B = h.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
#         h = torch.cat((cls_tokens, h), dim=1)

#         #---->Translayer x1
#         h = self.layer1(h) #[B, N, 512]

#         #---->PPEG
#         h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
#         #---->Translayer x2
#         h = self.layer2(h) #[B, N, 512]

#         #---->cls_token
#         h = self.norm(h)[:,0]

#         #---->predict
#         logits = self._fc2(h) #[B, n_classes]
#         Y_hat = torch.argmax(logits, dim=1)
#         Y_prob = F.softmax(logits, dim = 1)
#         results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
#         return results_dict

# if __name__ == "__main__":
#     data = torch.randn((1, 6000, 1536)).cuda()
#     model = TransMIL(n_classes=2).cuda()
#     print(model.eval())
#     results_dict = model(data = data)
#     print(results_dict)




# # """8moe1"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class PWLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(PWLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class MoEAdaptorLayer(nn.Module):
    def __init__(self, input_dim=1536, n_exps=3, layers=[1536, 1024], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(input_dim, layers[1], dropout) for _ in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(input_dim, n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate  
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        output = multiple_outputs.sum(dim=-2)
        return output

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=1024):
        super(TransLayer, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=1024):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.moe_adaptor = MoEAdaptorLayer(input_dim=1536, n_exps=3, layers=[1536, 1024], dropout=0.1, noise=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=1024)
        self.layer2 = TransLayer(dim=1024)
        self.pos_layer = PPEG(dim=1024)
        self.norm = nn.LayerNorm(1024)
        self._fc2 = nn.Linear(1024, n_classes)

    def forward(self, **kwargs):
        h = kwargs['data'].float()
        h = self.moe_adaptor(h)          
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1536)).cuda()
    model = TransMIL(n_classes=2).cuda()
    model.eval()
    print(model)
    results_dict = model(data=data)
    print(results_dict)





"""MLP"""
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from nystrom_attention import NystromAttention

# # 新增 MLP 模块
# class PreMLP(nn.Module):
#     def __init__(self, in_dim=1536, hidden_dim=512, out_dim=1024):
#         super(PreMLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear( hidden_dim, out_dim),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.mlp(x)



# class TransLayer(nn.Module):
#     def __init__(self, norm_layer=nn.LayerNorm, dim=512):
#         super().__init__()
#         self.norm = norm_layer(dim)
#         self.attn = NystromAttention(
#             dim=dim,
#             dim_head=dim // 8,
#             heads=8,
#             num_landmarks=dim // 2,
#             pinv_iterations=6,
#             residual=True,
#             dropout=0.1
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm(x))
#         return x


# class PPEG(nn.Module):
#     def __init__(self, dim=512):
#         super(PPEG, self).__init__()
#         self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
#         self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
#         self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         cls_token, feat_token = x[:, 0], x[:, 1:]
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
#         x = x.flatten(2).transpose(1, 2)
#         x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
#         return x


# class TransMIL(nn.Module):
#     def __init__(self, n_classes):
#         super(TransMIL, self).__init__()
#         self.pre_mlp = PreMLP(in_dim=1536, hidden_dim=512, out_dim=1024)
#         self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
#         self.pos_layer = PPEG(dim=512)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
#         self.n_classes = n_classes
#         self.layer1 = TransLayer(dim=512)
#         self.layer2 = TransLayer(dim=512)
#         self.norm = nn.LayerNorm(512)
#         self._fc2 = nn.Linear(512, self.n_classes)

#     def forward(self, **kwargs):
#         h = kwargs['data'].float()  # [B, n, 1536]
        
#         h = self.pre_mlp(h)         # 新增 MLP 处理
#         h = self._fc1(h)            # [B, n, 512]

#         # ----> pad
#         H = h.shape[1]
#         _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
#         add_length = _H * _W - H
#         h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

#         # ----> cls_token
#         B = h.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
#         h = torch.cat((cls_tokens, h), dim=1)

#         # ----> Translayer x1
#         h = self.layer1(h)  # [B, N, 512]

#         # ----> PPEG
#         h = self.pos_layer(h, _H, _W)  # [B, N, 512]

#         # ----> Translayer x2
#         h = self.layer2(h)  # [B, N, 512]

#         # ----> cls_token
#         h = self.norm(h)[:, 0]

#         # ----> predict
#         logits = self._fc2(h)  # [B, n_classes]
#         Y_hat = torch.argmax(logits, dim=1)
#         Y_prob = F.softmax(logits, dim=1)
#         results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
#         return results_dict


# if __name__ == "__main__":
#     data = torch.randn((1, 6000, 1536)).cuda()
#     model = TransMIL(n_classes=2).cuda()
#     print(model.eval())
#     results_dict = model(data=data)
#     print(results_dict)







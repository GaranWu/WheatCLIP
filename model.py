import torch
import clip
from backbone_vgg16 import *
from torch import nn
from fightingcv_attention.attention.CBAM import CBAMBlock


class mlp_block(nn.Module):
    def __init__(self, in_channels, mlp_dim, drop_ratio=0.):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            nn.Linear(mlp_dim, in_channels),
            nn.Dropout(drop_ratio)
        )

    def forward(self, x):
            x = self.block(x)
            return x

class mlp_layer(nn.Module):
    def __init__(self, seq_length_s, hidden_size_c, dc, ds, drop=0.):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size_c)
        # 注意两个block分别作用于输入的行和列， 即SXC，所以in_channels不一样
        self.token_mixing = mlp_block(in_channels=seq_length_s, mlp_dim=int(dc * seq_length_s), drop_ratio=drop)
        self.channel_mixing = mlp_block(in_channels=hidden_size_c, mlp_dim=int(ds * hidden_size_c), drop_ratio=drop)

    def forward(self, x):
        x1 = self.ln(x)
        x2 = x1.transpose(1, 2)  # 转置矩阵
        x3 = self.token_mixing(x2)
        x4 = x3.transpose(1, 2)

        y1 = x + x4  # skip-connection
        y2 = self.ln(y1)
        y3 = self.channel_mixing(y2)
        y = y1 + y3

        return y

# 按照paper中的 Table 1 来配置参数
class mlp_mixer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_num=4,
                 patch_size=32,
                 hidden_size_c=768,
                 seq_length_s=49,
                 dc=0.5,
                 ds=4,
                 drop=0.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.layer_num = layer_num
        self.hidden_size_c = hidden_size_c
        self.seq_length_s = seq_length_s
        self.dc = dc
        self.ds = ds

        self.ln = nn.LayerNorm(self.hidden_size_c)

        # 图片切割并做映射embedding，通过一个卷积实现
        self.proj = nn.Conv2d(self.in_channels, self.hidden_size_c, kernel_size=self.patch_size,
                              stride=self.patch_size)

        # 添加多个mixer-layer
        self.mixer_layer = nn.ModuleList([])
        for _ in range(self.layer_num):
            self.mixer_layer.append(mlp_layer(seq_length_s, hidden_size_c, ds, dc, drop))


    # 定义正向传播过程
    def forward(self, x):
        
        # flatten: [B, C, H, W] -> [B, C, HW]  # 第二个维度上展平 刚好是高度维度
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)

        for mixer_layer in self.mixer_layer:
            x = mixer_layer(x)
        x = self.ln(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim,exchange=False, align=True):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.align = align
        self.exchange = exchange

        # # Vit
        # self.linear = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.GELU()
        # )


    def forward(self, x, y):
        batch_size, C, width, height = x.size()
        if self.align:
        # Broadcasting and reshaping clip_features to match x dimensions
        #     y = self.linear(y)    #Vit
            y = y.view(batch_size, C, 1, 1)
            y = y.expand(batch_size, C, width, height)
        if self.exchange:
            q = x
            v = k = y
        else:
            q = y
            v = k = x
        # Self attention mechanism
        proj_query = self.query_conv(q).view(batch_size, -1, width * height)
        proj_key = self.key_conv(k).view(batch_size, -1, width * height).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(v).view(batch_size, -1, width * height)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class Multi_Granularity(nn.Module):
    def __init__(self,
                 layer_num = 4,
                 hidden_size_c = 256,
                 device = "cuda",
                 backbone_name = "vgg16",
                 seq_all = 336,
                 negative = False
                 ):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_size_c = hidden_size_c
        self.device = device
        self.mlp_mixer = nn.ModuleList([])
        self.mlp_layer = nn.ModuleList([])
        self.patch = [16, 8, 4, 32, 64]
        self.seq = [16, 64, 256, 256]
        self.seq_all = seq_all
        self.negative = negative
        self.backbone_name = backbone_name
        if self.backbone_name in ['vgg16', 'resnet50']:
            self.in_channel = 512
        elif self.backbone_name in ['mobilenet_v2']:
            self.in_channel = 32
        elif self.backbone_name in ['convnext_tiny']:
            self.in_channel = 192
        elif self.backbone_name in ['resnet34']:
            self.in_channel = 128
        else:
            raise AssertionError("backbone selection error")



        for i in range(3):
            self.mlp_mixer.append(mlp_mixer(in_channels=512, layer_num=self.layer_num, patch_size=self.patch[i],
                                            hidden_size_c=self.hidden_size_c, seq_length_s=self.seq[i]))

        # for _ in range(self.layer_num):
        #     self.mlp_layer.append(mlp_layer(hidden_size_c=self.hidden_size_c, seq_length_s= self.seq_all, dc=0.5, ds=4))



        self.linear = nn.Linear(self.seq_all, 256)

        self.CounterHead1 = nn.Sequential(
            nn.Conv2d(self.hidden_size_c, 128, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
        )
        self.CounterHead2 = nn.Sequential(
            nn.Conv2d(self.hidden_size_c, 128, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
        )

        self.CounterHead3 = nn.Sequential(
            nn.Conv2d(self.hidden_size_c, 128, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
        )
        self.CounterHead = nn.Sequential(
            nn.Conv2d(384,128,1),
            nn.ReLU(),
            nn.Conv2d(128, 64,1),
            nn.ReLU(),
            nn.Conv2d(64, 1,1),

        )





        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.backbone = Backbone(name=self.backbone_name)
        self.backbone.to(device=self.device)
        self.backbone.load_state_dict(torch.load("{}.pth".format(self.backbone_name)),strict=False)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.cbam = CBAMBlock(channel=self.in_channel, reduction=16, kernel_size=7)
        self.cbam.to(device=self.device)

        #clip
        clip_weight = "ViT-B/32"
        self.clip_model, self.preprocess = clip.load(clip_weight, device=self.device)
        print("clip_weight:{}".format(clip_weight))
        #
        # #Vit
        # self.Vit = timm.create_model("vit_base_patch16_224", pretrained=True)



        self.transformer1 = SelfAttention(dim=512)
        self.transformer1.to(device=self.device)


    def forward(self, input, sample):
        #经过vgg16
        feature = self.backbone(input)

        # sample 包含多个样本，其中第一个是正例，其他是负例
        batch_size, num_samples, c, h, w = sample.shape

        #clip
        with torch.no_grad():
            positive_sample = self.clip_model.encode_image(sample[:,0])
            if self.device == "cuda":
                positive_sample = positive_sample.type(torch.cuda.FloatTensor)
            if self.negative:
                negative_samples = [self.clip_model.encode_image(sample[:, i]) for i in range(1, num_samples)]
                if self.device == "cuda":
                    negative_samples = [s.type(torch.cuda.FloatTensor) for s in negative_samples]


        x = self.transformer1(feature, positive_sample)
        if self.negative:
            negative_samples = [self.transformer1(feature, ns) for ns in negative_samples]
            negative_samples = torch.stack(negative_samples)
            positive_sample = x.unsqueeze(0)
            contrast_feature = torch.cat([positive_sample,negative_samples],dim=0)

        # #Vit
        # with torch.no_grad():
        #     positive_sample = self.Vit.forward_features(sample[:,0])[:,0,:]
        #     if self.device == "cuda":
        #         positive_sample = positive_sample.type(torch.cuda.FloatTensor)
        #     if self.negative:
        #         negative_samples = [self.Vit.forward_features(sample[:, i])[:,0,:] for i in range(1, num_samples)]
        #         if self.device == "cuda":
        #             negative_samples = [s.type(torch.cuda.FloatTensor) for s in negative_samples]

        #
        # x = self.transformer1(feature, positive_sample)
        # if self.negative:
        #     negative_samples = [self.transformer1(feature, ns) for ns in negative_samples]
        #     negative_samples = torch.stack(negative_samples)
        #     positive_sample = x.unsqueeze(0)
        #     contrast_feature = torch.cat([positive_sample,negative_samples],dim=0)



        x = self.cbam(feature)
        b = x.shape[0]
        c = 256
        x_coarse = self.mlp_mixer[0](x).permute(0,2,1).reshape(b,c,self.patch[2],self.patch[2])
        x_middle = self.mlp_mixer[1](x).permute(0,2,1).reshape(b,c,self.patch[1],self.patch[1])
        x_fine = self.mlp_mixer[2](x).permute(0,2,1).reshape(b,c,self.patch[0],self.patch[0])
        # x_all = torch.cat([ x_coarse, x_middle, x_fine], 1)
        # for mlp_layer in self.mlp_layer:
        #     x_all = mlp_layer(x_all)
        #
        # b, hw, c = x_all.shape
        # x_coarse = x_all[:, :16, :].permute(0,2,1).reshape(b,c,self.patch[2],self.patch[2])
        # x_middle = x_all[:, 16:80, :].permute(0,2,1).reshape(b,c,self.patch[1],self.patch[1])
        # x_fine = x_all[:, 80:, :].permute(0,2,1).reshape(b,c,self.patch[0],self.patch[0])


        x_coarse = self.CounterHead1(x_coarse)
        x_middle = self.CounterHead2(x_middle)
        x_fine = self.CounterHead3(x_fine)
        x_all = torch.cat([x_coarse, x_middle, x_fine], 1)
        x_count = self.CounterHead(x_all)
        if self.negative:
            return x_count,contrast_feature
        else:
            return x_count


if __name__ == '__main__':
    neck = Multi_Granularity(device="cpu", layer_num=4,negative=True)
    # neck.load_state_dict(torch.load("mobilenetv2_1.0-f2a8633.pth", map_location="cpu"))
    input = torch.ones((16, 3, 512, 512))
    clip_image = torch.ones((16,11,3,224,224 ))
    # summary(neck, (3, 512, 512))
    # imgs = torch.ones((16, 3, 512, 512))
    output,contrast_feature = neck(input,clip_image)
    print(output.shape,contrast_feature.shape)
    # print(neck)


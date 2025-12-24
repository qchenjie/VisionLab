import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math

def window_partition(x, window_szie = 8):
    b, c, h, w = x.shape

    pad = [ 0, 0, 0, 0 ]
    if h % window_szie != 0:
        pad[ 3 ] = window_szie - h % window_szie

    if w % window_szie != 0:
        pad[ 1 ] = window_szie - w % window_szie

    output = F.pad(x, pad = pad, mode = 'reflect')
    *_, H, W = output.shape

    output = output.contiguous().view(b, c, window_szie, H // window_szie, window_szie, W // window_szie)
    output = output.permute((0, 1, 2, 4, 3, 5)).contiguous().view(b, c * window_szie * window_szie, H // window_szie,
                                                                  W // window_szie)

    return output, h, w, c, H // window_szie


def window_reverse(x = None, inplanes = None, window_size = 8, patch = None):
    b, l, d = x.shape
    # TODO? BUG
    x = x.view((b, patch, l // patch, inplanes, window_size, window_size))
    x = x.contiguous().permute((0, 3, 4, 1, 5, 2))

    H = x.shape[ 2 ] * x.shape[ 3 ]
    x = x.contiguous().view(b, inplanes, H, -1)

    return x


class cross_attention(nn.Module):
    def __init__(self,
                 inplanes, window_size = 8):
        super(cross_attention, self).__init__()

        c = inplanes * window_size * window_size

        self.inplanes = c

    def forward(self,
                q = None,
                k = None,
                v = None):
        b, *_ = q.shape

        # TODO?

        q, qh, qw, qc, patch = window_partition(q, window_szie = 8)

        q = q.contiguous().permute((0, 2, 3, 1)).contiguous().view(b, -1, self.inplanes)

        k, kh, kw, kc, _ = window_partition(k, window_szie = 8)
        k = k.contiguous().permute((0, 2, 3, 1)).contiguous().view(b, -1, self.inplanes)

        v, vh, vw, vc, _ = window_partition(v, window_szie = 8)
        v = v.contiguous().permute((0, 2, 3, 1)).contiguous().view(b, -1, self.inplanes)

        _q = F.normalize(q, p = 2)
        _k = F.normalize(k, p = 2)
        _v = F.normalize(v, p = 2)

        similarity = _q @ _k.transpose(-2, -1) * (self.inplanes ** (-0.5))


        sim = F.softmax(similarity, dim = -1) @ _v

        out = window_reverse(x = sim, inplanes = qc, patch = patch)

        return out[ :, :, :qh, :qw ]

class CBA(nn.Module):
    def __init__(self,in_planes = None,
                 out_planes = None,
                 ksize = 3,
                 stride = 1,
                 dilation = 1
                 ):
        super(CBA, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels = in_planes,
                              out_channels = out_planes,
                              kernel_size = ksize,
                              padding = ksize // 2,
                              stride = stride,dilation = dilation),
                                  nn.BatchNorm2d(out_planes),
                                  nn.LeakyReLU(inplace = True),
                                  nn.Conv2d(in_channels = out_planes,
                                            out_channels = out_planes,
                                            kernel_size = 1))
    def forward(self,x):
        out = self.layer(x)
        return out

class ResBlock(nn.Module):
    def __init__(self,in_planes = None,
                 out_planes = None,
                 ksize = 3,
                 stride = 1,
                 num_blocks = 3):
        super(ResBlock, self).__init__()

        self.layer = nn.ModuleList([])
        for i in range(num_blocks):
           self.layer.append(CBA(in_planes = in_planes if i==0 else out_planes,
                                             out_planes = out_planes,
                                             ksize = ksize,
                                             stride = stride))

        self.shortcut = ((in_planes == out_planes) and (stride == 1))
    def forward(self,x):
        x1 = x
        for ind,layer in enumerate(self.layer):
          x = layer(x)
        return x + x1 if self.shortcut else x


class FeedForward(nn.Module):
    def __init__(self, dim = None, mult=4):
        super().__init__()

        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(

            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )


        self.gate = nn.Sequential(
                    nn.MaxPool2d(kernel_size = 3,padding = 1,stride = 1),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)
        out = self.net(x)

        gate = self.gate(x)

        return x * gate + out

class SEB(nn.Module):
    def __init__(self,in_planes = None,
              out_planes = None):
        super(SEB, self).__init__()
        # TODO? 进行多层级特征感知交互

        self.alpha = nn.Parameter(torch.tensor(0.2),requires_grad = True)
        self.layer1 = nn.Sequential(
                      nn.Conv2d(in_channels = in_planes,
                                out_channels = out_planes,kernel_size = 3,
                                padding = 1),
                      nn.BatchNorm2d(out_planes),
                      nn.AdaptiveAvgPool2d(1),
                      nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(ResBlock(in_planes = in_planes,
                                     out_planes = out_planes,
                                     ksize = 3),
                                    FeedForward(dim = out_planes)
                                    )

        self.fin = nn.Sequential(nn.Conv2d(in_channels = 2 * out_planes,
                             out_channels = out_planes,
                             kernel_size = 3,
                             padding = 1),
                                 nn.BatchNorm2d(out_planes),
                                 nn.LeakyReLU(inplace = True),
                                 nn.Conv2d(in_channels = out_planes,
                             out_channels = out_planes,
                             kernel_size = 1))
    def forward(self,x):

        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out = self.alpha * out1 + out2
        out = torch.cat((out,out2),dim = 1)
        out = self.fin(out)
        return out


class FrequencySpace(nn.Module):
    def __init__(self,in_planes = None,
                 out_planes = None):
        super(FrequencySpace, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels = in_planes,
                              out_channels = in_planes,kernel_size = 3,padding = 1),
                                  ResBlock(in_planes = in_planes,out_planes = in_planes,
                                           ksize = 1))
        self.out = CBA(in_planes = 1,out_planes = in_planes,ksize = 1)

        self.amp = nn.Sequential(SEB(in_planes = in_planes,
                       out_planes = out_planes),
                                 FeedForward(out_planes),
                                 nn.Conv2d(in_channels = out_planes,
                                           out_channels = in_planes, kernel_size = 3, padding = 1)
                                 )

        self.amp_conv = CBA(in_planes = 2 * in_planes, out_planes = out_planes)
        self.phase_conv = CBA(in_planes = 2 * in_planes,out_planes = out_planes)

    def forward(self,x):
        B,C,h,w = x.shape

        out = fft.rfft2(x)
        amp = torch.abs(out)
        phase1 = torch.angle(out)
        # TODO? 对细节进行提取并且获取细节差异性进行补偿交互
        phase2 = self.conv(phase1)

        phase_cos1 = phase1.contiguous().view((B, C, -1))  # (B, C, H*W)
        phase_cos2 = phase2.contiguous().view((B, C, -1))  # (B, C, H*W)
        phase1_norm = F.normalize(phase_cos1,dim = 1,p = 2)
        phase2_norm = F.normalize(phase_cos2,dim = 1,p = 2)
        similarity_matrix = torch.bmm(phase1_norm, phase2_norm.permute((0, 2, 1)))  # (B, C, C)

        similarity_scores = similarity_matrix.mean(dim = -1)  # (B, C)

        prob = torch.argmax(similarity_scores, dim = -1)  # (B,)
        # 从每个batch中提取相似度最高的通道
        mask = torch.stack([ phase2[ b, prob[ b ] ] for b in range(B) ]).unsqueeze(dim = 1)  # (B, 1, H, W)

        phase_out = F.sigmoid(self.out(mask)) * phase1  # (B, C, H, W)

        phase_out = self.phase_conv(torch.cat((phase_out,phase2),dim = 1))

        amp_out = self.amp_conv(torch.cat((self.amp(amp),amp),dim = 1))

        real = amp_out * torch.cos(phase_out)
        imag = amp_out * torch.sin(phase_out)

        out = torch.complex(real,imag)
        out = fft.irfft2(out)
        return out


#TODO? 噪声抑制
class NoiseSuppress(nn.Module):
    def __init__(self,in_planes = None,
                 out_planes = None,
                 k = 2):
        super(NoiseSuppress, self).__init__()

        self.k = k #TODO? 特征聚类
        self.conv = nn.Sequential(nn.Conv2d(in_channels = in_planes,out_channels = out_planes,
                              kernel_size = 3,padding = 1),
                                  nn.LeakyReLU(inplace = True),
                                  )

        self.fre = FrequencySpace(in_planes = in_planes,out_planes = out_planes)
        self.spa = SEB(in_planes = in_planes,out_planes = out_planes)

        self.cross = cross_attention(inplanes = out_planes)
        #TODO? 查询语义一致性
        self.layer1 = nn.Sequential(CBA(in_planes = 1 + self.k,out_planes =
                          out_planes,ksize = 3),
                                    nn.Conv2d(in_channels = out_planes,
                                              out_channels = out_planes,
                                              kernel_size = 1))

        self.layer2 = nn.Sequential(CBA(in_planes = 2 * out_planes, out_planes =
        out_planes, ksize = 3),
                                    nn.Conv2d(in_channels = out_planes,
                                              out_channels = out_planes,
                                              kernel_size = 1))

        self.scale = nn.Parameter(torch.tensor((1.),requires_grad = True))
        self.shift = nn.Parameter(torch.tensor((0.),requires_grad = True))

        self.shortcut = (in_planes == out_planes)
    def forward(self,x = None):
        uncertainty = self._uncertanity_map(x)
        # TODO?
        fre = self.fre(x)
        spa = self.spa(x)

        attention = self.cross(fre,spa,fre)

        samentic = (F.normalize(spa,dim = 1) - F.normalize(attention,dim = 1)) ** 2
        _,index = torch.topk(samentic,dim = 1,k = self.k)
        mask = torch.gather(x,dim = 1,index = index) #TODO?

        #TODO?
        out = self.layer1(torch.cat((uncertainty,mask),dim = 1))
        out1 = self.layer2(torch.cat((fre,spa),dim = 1))

        output1 = (self.scale * out + out1) / (F.sigmoid(self.shift) + 1e-10)

        return x + output1 if self.shortcut else output1

    def _uncertanity_map(self,map):
        b,c,h,w = map.shape
        prob = F.sigmoid(map)

        prob = -torch.sum(prob * torch.log(prob + 1e-6),dim = 1,keepdim = True)
        prob /= math.log(c)
        return prob

# TODO? copy-paste方法进行像素级实例建模

# TODO? 基于边界感知的mamba扫描机制

class Masked(nn.Module):
    def __init__(self,ksize = 3,
                 mask = 0.25,
                 stride = 1,
                 in_planes = None,
                 out_planes = None,
                 H = None):
        super(Masked, self).__init__()
        self.patch = nn.Unfold(kernel_size = ksize)
        self.num_patch = (H // stride - ksize + 1) ** 2
        hidden = int(mask * self.num_patch)

        self.ffn = FeedForward(dim = in_planes)
        self.layer = nn.Sequential(
                     nn.Linear(self.num_patch,hidden),
                     nn.GELU(),
                     nn.LayerNorm(hidden),
                     nn.Linear(hidden,hidden)
        )

        self.conv = nn.Sequential(
                    nn.LayerNorm(self.num_patch),
                    nn.Linear(self.num_patch, self.num_patch)
        )

        self.mask = mask
        self.hidden = hidden

        self.fold = nn.Fold(output_size = (H // stride,H // stride),kernel_size = ksize)

        self.fin = nn.Sequential(
                      ResBlock(in_planes = 2 * in_planes,out_planes = out_planes),
                      nn.Conv2d(in_channels = out_planes,
                                out_channels = out_planes,
                                kernel_size = 1)
                      )

    def forward(self,x):
        x1 = self.ffn(x)

        out = self.patch(x1)#[b,c,h,w]->[b,c*num,h,w]
        out1 = self.layer(out)

        unmask = self.num_patch - self.hidden

        _,ind = torch.topk(out,dim = -1,k = unmask)

        out2 = torch.gather(out,dim = -1,index = ind)
        output1 = torch.cat((out1,out2),dim = -1)

        out = self.conv(output1)
        out = self.fold(out)
        out = self.fin(torch.cat((x,out),dim = 1))
        return out


class UP(nn.Module):
    def __init__(self,in_planes1,in_planes2,out_planes):
        super(UP, self).__init__()
        self.up = nn.Upsample(scale_factor = 2,mode = 'bilinear',align_corners = True)
        self.conv1 = nn.Conv2d(in_channels = in_planes1 + in_planes2,out_channels = out_planes,kernel_size = 3,padding = 1)
        self.relu1 = nn.LeakyReLU(inplace = True)

        self.out = nn.Sequential(nn.Conv2d(in_channels = out_planes, out_channels = out_planes, kernel_size = 3, padding = 1),
                                 nn.LeakyReLU(inplace = True))

    def forward(self,x,x1):
        #--------------------------#
        # x1降采样
        #--------------------------#
        outputs = torch.cat((x,self.up(x1)),dim = 1)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        return self.out(outputs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft
import os,sys
sys.path.insert(0,os.getcwd())
from nets.BaMamba.block_util import cross_attention,CBA,ResBlock,SEB,\
                               FeedForward,FrequencySpace,NoiseSuppress,Masked,UP
from nets.BaMamba.ss2d import BoundaryMamaba

class BMENCODER(nn.Module):
    def __init__(self,in_planes = None,
                 hidden = 64,
                 depth = [3,4,5,2],
                 width = [1.5,1.,1.5,2.],
                 size = (256,256)):
        super(BMENCODER, self).__init__()

        self.hidden = hidden
        self.conv = CBA(in_planes = in_planes,out_planes = self.hidden ,ksize = 5,stride = 2)

        self.stage1 = self._make_layer(self.hidden,int(self.hidden * width[0]),stride = 1,num_blocks = depth[0])
        self.stage2 = self._make_layer(int(self.hidden * width[0]),int(self.hidden * width[1]),stride = 2,num_blocks = depth[1])

        self.Masked1 = Masked(in_planes = int(self.hidden * width[1]), out_planes = int(self.hidden * width[1]),
                             stride = 4, H = size[ 0 ])
        self.stage3 = self._make_layer(int(self.hidden * width[1]), int(self.hidden * width[2]), stride = 2, num_blocks = depth[2])

        self.boundarymamaba1 = BoundaryMamaba(in_planes =int(self.hidden * width[2]),num_blocks = 3 )
        self.noise1 = NoiseSuppress(in_planes = int(self.hidden * width[2]),out_planes = int(self.hidden * width[2]),
                                    k = 5)
        self.stage4 = self._make_layer(int(self.hidden * width[2]), int(self.hidden * width[3]), stride = 2, num_blocks = depth[3])
        # self.Masked2 = Masked(in_planes = int(self.hidden * width[ 3 ]), out_planes = int(self.hidden * width[ 3 ]),
        #                       stride = 16, H = size[ 0 ])
    def _make_layer(self,
                    in_planes = None,
                    out_planes = None,
                    stride = 1,
                    num_blocks = None):

        layers = [ ]
        if stride != 1:
         downsample = CBA(in_planes = in_planes,
                         out_planes = out_planes,
                          ksize = 3,
                          stride = stride)
         layers.append(downsample)

        layers.append(
                ResBlock(in_planes = out_planes if stride != 1 else in_planes,out_planes = out_planes,
                         ksize = 1,num_blocks = num_blocks),

            )
        # TODO?
        # layers.append(NoiseSuppress(in_planes = out_planes,out_planes = out_planes,
        #               k = 5))

        return nn.Sequential(*layers)

    def forward(self,x):

        x1 = self.conv(x)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x2 = self.Masked1(x2)

        x3 = self.stage3(x2)

        x3 = self.noise1(x3)
        x3 = self.boundarymamaba1(x3)
        x4 = self.stage4(x3)
        #x4 = self.Masked2(x4)

        return [x1,x2,x3,x4]

class BMDECODER(nn.Module):
    def __init__(self,hidden = 64,
                 width = [ 1.5, 1., 1., 2. ],
                 classfier = 8
                 ):
        super(BMDECODER, self).__init__()

        self.up1 = UP(int(hidden * width[ 2 ]), int(hidden * width[ 3 ]), int(hidden * width[ 3 ]))
        self.denoise1 = NoiseSuppress(in_planes = int(hidden * width[ 3 ]),out_planes = int(hidden * width[ 3 ]))
        self.up2 = UP(int(hidden * width[ 3 ]), int(hidden * width[ 1 ]), int(hidden * width[ 1 ]))
        self.denoise2 = NoiseSuppress(in_planes = int(hidden * width[ 1 ]),out_planes = int(hidden * width[ 1 ]))
        self.up3 = UP(int(hidden * width[ 1 ]), int(hidden * width[ 0 ]), int(hidden * width[ 0 ]))
        self.boundarydecoder1 = BoundaryMamaba(in_planes = int(hidden * width[0]),num_blocks = 6)
        self.up4 = nn.Sequential(
                   nn.ConvTranspose2d(in_channels = int(hidden * width[ 0 ]),out_channels = int(hidden * width[ 0 ] * 2.),kernel_size = 4,padding = 1,stride = 2,dilation = 1),
                   CBA(in_planes = int(hidden * width[ 0 ] * 2),out_planes = int(hidden * width[ 0 ])),
                   nn.Conv2d(in_channels = int(hidden * width[ 0 ]),out_channels = int(hidden * width[ 0 ]),kernel_size = 3,padding = 1),
                   nn.LeakyReLU(inplace = True)
        )
        self.boundarydecoder2 = BoundaryMamaba(in_planes = int(hidden * width[0]),num_blocks = 4)

        self.fin = nn.Sequential(nn.Conv2d(in_channels = int(hidden * width[ 0 ]),out_channels = classfier,kernel_size = 3,padding = 1),
                                 )

        self.boundary = nn.Sequential(nn.Conv2d(in_channels = int(hidden * width[ 0 ]),out_channels = int(hidden * width[ 0 ]) // 2,kernel_size = 3,padding = 1),
                                      nn.BatchNorm2d(int(hidden * width[ 0 ]) // 2),
                                      nn.Conv2d(in_channels = int(hidden * width[ 0 ]) // 2,
                                                out_channels = int(hidden * width[ 0 ]) // 2, kernel_size = 1),
                                      nn.Conv2d(in_channels = int(hidden * width[ 0 ]) // 2,
                                                out_channels = 1, kernel_size = 3,padding = 1),                                 )

    def forward(self,feats):
        x1,x2,x3,x4 = feats

        out1 = self.up1(x3,x4)
        out1 = self.denoise1(out1)
        out2 = self.up2(x2,out1)
        out2 = self.denoise2(out2)
        out3 = self.up3(x1,out2)
        out3 = self.boundarydecoder1(out3)
        output = self.up4(out3)

        output = self.boundarydecoder2(output)
        #TODO? 
        boundary = F.sigmoid(self.boundary(output))

        return self.fin(output),boundary


class BaMamba(nn.Module):
    def __init__(self,in_planes = None,
                 out_planes = None,
                 size = (256,256),
                 hidden = 64,
                 depth = [ 3, 4, 5, 2 ],
                 width = [ 1.5, 1., 1.5, 2. ],
                 *args,
                 **kwargs
    ):
        super(BaMamba, self).__init__()

        self.encoder = BMENCODER(in_planes = in_planes,
                                 hidden = hidden,
                                 depth = depth,
                                 width = width,
                                 size = size)

        self.decoder = BMDECODER(hidden = hidden,width = width,classfier = out_planes)

    def forward(self,x):

        outputs = self.encoder(x)
        pred,boundary = self.decoder(outputs)
        return pred,boundary


if __name__ == '__main__':
    x = torch.randn((1,3,256,256)).cuda()
    model = BaMamba(in_planes = 3,out_planes = 8).cuda()
    #model = NoiseSuppress(in_planes = 16,out_planes = 8)
    # model = Masked(ksize = 5,H = 256,in_planes = 128,out_planes = 256,stride = 16).cuda()
    #
    # out = model(x)
    # print(out.shape)

    # model = BMENCODER(in_planes = 3).cuda()
    #
    # out = model(x)
    # for o in out:
    #     print(o.shape)

    # model = BMDECODER(hidden = 64)
    # feats = [torch.randn((3,96,128,128)),
    #          torch.randn((3,64,64,64)),
    #          torch.randn((3,64,32,32)),
    #          torch.randn((3,128,16,16))]
    #
    out,b = model(x)
    print(out.shape,b.shape)

    # from thop import profile
    # flops,params = profile(model,(1,3,256,256))
    # print(f'FLOPs:{flops /1e9}   Params:{params / 1e6}')
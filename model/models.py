import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def upsample(src, tar):
    src = F.upsample(src, size= tar.shape[2:], mode='bilinear')
    return(src)
'''In U2NET we save params by upsampling instead of convTranspose2d '''

class RBC(nn.Module):
    def __init__(self, in_ch, out_ch, dirate = 1):
        super(RBC, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1*dirate, dilation = 1*dirate)
        self.ReLu = nn.ReLU(True)
        self.batch_norm = nn.BatchNorm2d(out_ch)
    def forward(self,x):
        hx = x
        x_out = self.ReLu(self.batch_norm(self.conv1(x)))
        return x_out
    

class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.convin = RBC(3, 64)
        self.rbc1 = RBC(64, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64,32)
        self.rbc1d = RBC(64,64)
    
    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)


        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx5h = self.maxpool56(hx5)

        hx6 = self.rbc6(hx5h)
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))
        
        return hx1d + hxin
    
class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.rbcin = RBC(64, 128)
        self.rbc1 = RBC(128, 64)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(64,64)
        self.maxpoool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True)
        self.rbc3 = RBC(64, 64)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(64, 64)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode = True)
        self.rbc5 = RBC(64, 64)
        self.rbc6 = RBC(64, 64, dirate = 2)

        self.rbc5d = RBC(128, 64)
        self.rbc4d = RBC(128, 64)
        self.rbc3d = RBC(128, 64)
        self.rbc2d = RBC(128, 64)
        self.rbc1d = RBC(128, 128)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpoool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx6 = self.rbc6(hx5)

        hx5d = self.rbc5d(torch.cat((hx5, hx6), 1))
        hx = upsample(hx5d, hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)

        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))

        return hxin + hx1d

class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.rbcin = RBC(128, 256)
        self.rbc1 = RBC(256, 128, dirate = 2)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(128, 128, dirate = 2)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(128, 128, dirate = 2)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(128, 128, dirate = 2)
        self.rbc5 = RBC(128, 128, dirate = 2)

        self.rbc4d = RBC(256, 128, dirate =2)
        self.rbc3d = RBC(256, 128, dirate =2)
        self.rbc2d = RBC(256, 128, dirate =2)
        self.rbc1d = RBC(256, 256, dirate =2)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)
        
        hx4 = self.rbc4(hx3h)
        hx5 = self.rbc5(hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx5), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)

        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))

        return hxin + hx1d

class Block4(nn.Module):
    def __init__(self):
        super(Block4, self).__init__()
        self.rbcin = RBC(256, 512)
        self.rbc1 = RBC(512, 256, dirate=3)
        self.maxpol12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(256, 256, dirate= 3)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(256, 256, dirate=3)
        self.rbc4 = RBC(256, 256, dirate=3)

        self.rbc3d = RBC(512, 256, dirate = 3)
        self.rbc2d = RBC(512, 256, dirate = 3)
        self.rbc1d = RBC(512, 512, dirate = 3)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpol12(hx1)
        hx2  = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)
        hx3 = self.rbc3(hx2h)
        hx4 = self.rbc4(hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx4), 1))
        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))

        return hxin + hx1d
    
class Block5(nn.Module):
    def __init__(self):
        super(Block5, self).__init__()
        self.rbc1 = RBC(512, 1024)
        self.rbc2 = RBC(1024, 512)
        self.conv1 = nn.Conv2d(512, 1, kernel_size = 3, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(1)
    
    def forward(self, x):
        hx = x
        x_model = self.rbc2(self.rbc1(hx))
        x_output = self.batch_norm1(self.conv1(x_model))

        return {'model_input': x_model, 'model_output':x_output}
    
class Block6(nn.Module):
    def __init__(self):
        super(Block6, self).__init__()
        self.rbcin = RBC(1024, 256)
        self.rbc1 = RBC(256, 256, dirate=3)
        self.maxpol12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(256, 256, dirate= 3)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(256, 256, dirate=3)
        self.rbc4 = RBC(256, 256, dirate=3)

        self.rbc3d = RBC(512, 256, dirate = 3)
        self.rbc2d = RBC(512, 256, dirate = 3)
        self.rbc1d = RBC(512, 256, dirate=3)
        self.c1d = nn.Conv2d(256, 1, kernel_size = 3, padding =3, dilation = 3)
        self.b1d = nn.BatchNorm2d(1)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpol12(hx1)
        hx2  = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)
        hx3 = self.rbc3(hx2h)
        hx4 = self.rbc4(hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx4), 1))
        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx1d_input = self.rbc1d(torch.cat((hx1, hx), 1))
        hx1d_output = self.b1d(self.c1d(hx1d_input))

        return {'model_input': hx1d_input+hxin, 'model_output':hx1d_output}

        
class Block7(nn.Module):
    def __init__(self):
        super(Block7, self).__init__()
        self.rbcin = RBC(512, 128)
        self.rbc1 = RBC(128, 128, dirate = 2)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(128, 128, dirate = 2)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(128, 128, dirate = 2)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(128, 128, dirate = 2)
        self.rbc5 = RBC(128, 128, dirate = 2)

        self.rbc4d = RBC(256, 128, dirate =2)
        self.rbc3d = RBC(256, 128, dirate =2)
        self.rbc2d = RBC(256, 128, dirate =2)
        self.rbc1d = RBC(256, 128, dirate =2)
        self.c1d = nn.Conv2d(128, 1, kernel_size = 3, padding =2, dilation = 2)
        self.b1d = nn.BatchNorm2d(1)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)
        
        hx4 = self.rbc4(hx3h)
        hx5 = self.rbc5(hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx5), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx1d_input = self.rbc1d(torch.cat((hx1, hx), 1))
        hx1d_output = self.b1d(self.c1d(hx1d_input))

        return {'model_input': hx1d_input+hxin, 'model_output':hx1d_output}
    

class Block8(nn.Module):
    def __init__(self):
        super(Block8, self).__init__()
        self.rbcin = RBC(256, 64)
        self.rbc1 = RBC(64, 64)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(64,64)
        self.maxpoool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True)
        self.rbc3 = RBC(64, 64)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(64, 64)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode = True)
        self.rbc5 = RBC(64, 64)
        self.rbc6 = RBC(64, 64, dirate = 2)

        self.rbc5d = RBC(128, 64)
        self.rbc4d = RBC(128, 64)
        self.rbc3d = RBC(128, 64)
        self.rbc2d = RBC(128, 64)
        self.rbc1d = RBC(128, 64)
        self.c1d = nn.Conv2d(64, 1, kernel_size = 3)
        self.b1d = nn.BatchNorm2d(1)
        


    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpoool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx6 = self.rbc6(hx5)

        hx5d = self.rbc5d(torch.cat((hx5, hx6), 1))
        hx = upsample(hx5d, hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        
        hx1d_input = self.rbc1d(torch.cat((hx1, hx), 1))
        hx1d_output = self.b1d(self.c1d(hx1d_input))

        return {'model_input': hx1d_input+hxin, 'model_output':hx1d_output}
    

class Block9(nn.Module):
    def __init__(self):
        super(Block9, self).__init__()
        self.convin = RBC(128, 1)
        self.rbc1 = RBC(1, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64, 32)
        self.c1d = nn.Conv2d(64, 1, kernel_size = 3, padding=1)
        self.b1d = nn.BatchNorm2d(1)
        self.ReLu = nn.ReLU(True)
        
    
    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)


        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx5h = self.maxpool56(hx5)

        hx6 = self.rbc6(hx5h)
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))
        hx1d_input = self.ReLu(hx1d_output)

        return {'model_input': hx1d_input, 'model_output':hx1d_output}
        
     
    

class U2NET(nn.Module):
    def __init__(self):
        super(U2NET, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.block1 = Block1() 
        self.block2 = Block2()
        self.block3 = Block3()
        self.block4 = Block4()
        self.block5 = Block5()
        self.block6 = Block6()
        self.block7 = Block7()
        self.block8 = Block8()
        self.block9 = Block9()
        self.outconv5 = nn.Conv2d(4, 1, kernel_size=3, padding =1)

    def forward(self, x):
        hx = x
        hx1 = self.block1(hx) #block1 output as a tensor
        hx = self.maxpool(hx1) # half
        hx2 = self.block2(hx) # 
        hx = self.maxpool(hx2)
        hx3 = self.block3(hx)
        hx = self.maxpool(hx3)
        hx4 = self.block4(hx)
        hx = self.maxpool(hx4)
        hx5 = self.block5(hx) #block 5 se outputs are dictionaries
        hx = upsample(hx5['model_input'], hx4)
        hx6 = self.block6(torch.cat((hx4, hx), 1))
        hx = upsample(hx6['model_input'], hx3)
        hx7 = self.block7(torch.cat((hx3, hx), 1))
        hx = upsample(hx7['model_input'], hx2)
        hx8 = self.block8(torch.cat((hx2, hx), 1))
        hx = upsample(hx8['model_input'], hx1)
        hx9 = self.block9(torch.cat((hx1, hx), 1))

        d9 = hx9['model_output']
        d = hx8['model_output']
        d8 = upsample(d, d9)
        d = hx7['model_output']
        d7 = upsample(d, d9)
        d = hx6['model_output']
        d6 = upsample(d, d9)
        d = hx5['model_output']
        d5 = upsample(d, d9)

        d0 = self.outconv5(torch.cat((d5, d6, d7, d8, d9),1))


        return F.sigmoid(d0), F.sigmoid(d9), F.sigmoid(d8), F.sigmoid(d7), F.sigmoid(d6), F.sigmoid(d5)



class Block1_new(nn.Module):
    def __init__(self):
        super(Block1_new, self).__init__()
        self.convin = RBC(2, 64)
        self.rbc1 = RBC(64, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64,32)
        self.rbc1d = RBC(64,64)
    
    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)


        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx5h = self.maxpool56(hx5)

        hx6 = self.rbc6(hx5h)
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))
        
        decoder_outputs = {'6d': hx6d, '5d':hx5d, '4d':hx4d, '3d':hx3d, '2d':hx2d, 'output':hx1d+hxin}

        return decoder_outputs
    

class Block9_new(nn.Module):
    def __init__(self):
        super(Block9_new, self).__init__()
        self.convin = RBC(64, 1)
        self.rbc1 = RBC(1, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(64, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(64, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(64,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(64,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(64, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64, 32)
        self.c1d = nn.Conv2d(64, 1, kernel_size = 3, padding=1)
        self.b1d = nn.BatchNorm2d(1)
        self.ReLu = nn.ReLU(True)
        
    
    def forward(self, x):
        x_skip = x
        hx = x_skip['output']
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1) #144 hx1h


        hx2 = self.rbc2(torch.cat((hx1h, x_skip['2d']), 1))   
        hx2h = self.maxpool23(hx2) #72

        hx3 = self.rbc3(torch.cat((hx2h, x_skip['3d']), 1))
        hx3h = self.maxpool34(hx3) #36

        hx4 = self.rbc4(torch.cat((hx3h, x_skip['4d']), 1))
        hx4h = self.maxpool45(hx4) #18

        hx5 = self.rbc5(torch.cat((hx4h, x_skip['5d']), 1))
        hx5h = self.maxpool56(hx5) #9

        hx6 = self.rbc6(torch.cat((hx5h, x_skip['6d']), 1))
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d_output = self.ReLu(self.b1d(self.c1d(torch.cat((hx1, hx), 1))))

        return hx1d_output
    
class UUNET(nn.Module):
    def __init__(self):
        super(UUNET, self).__init__()
        self.block1 = Block1_new()
        self.block9 = Block9_new()

    def forward(self, x):
        hx = x
        hx1= self.block1(hx)
        hx2 = self.block9(hx1)

        return F.sigmoid(hx2)



class Block9_new2(nn.Module):
    def __init__(self, num_classes=6):
        super(Block9_new2, self).__init__()
        self.num_classes = num_classes
        self.convin = RBC(64, 1)
        self.rbc1 = RBC(1, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64, 32)
        self.c1d = nn.Conv2d(64, self.num_classes, kernel_size = 3, padding=1)
        self.b1d = nn.BatchNorm2d(self.num_classes)
        self.ReLu = nn.ReLU(True)
        
    
    def forward(self, x):
        x_skip = x
        hx = x_skip['output']
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1) #144 hx1h


        hx2 = self.rbc2(hx1h + x_skip['2d'])   
        hx2h = self.maxpool23(hx2) #72

        hx3 = self.rbc3(hx2h+ x_skip['3d'])
        hx3h = self.maxpool34(hx3) #36

        hx4 = self.rbc4(hx3h+ x_skip['4d'])
        hx4h = self.maxpool45(hx4) #18

        hx5 = self.rbc5(hx4h+ x_skip['5d'])
        hx5h = self.maxpool56(hx5) #9

        hx6 = self.rbc6(hx5h+ x_skip['6d'])
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))

        return hx1d_output
    
class UU2NET(nn.Module):
    def __init__(self):
        super(UU2NET, self).__init__()
        self.block1 = Block1_new()
        self.block9 = Block9_new2()

    def forward(self, x):
        hx = x
        hx1= self.block1(hx)
        hx2 = self.block9(hx1)

        return F.softmax(hx2, dim=1)
    



def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)
    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
    

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        # print(h_x, h_y, w_x, w_y)
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N

        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return (mean_A * x + mean_b).float()
    



class Block1_3(nn.Module):
    def __init__(self):
        super(Block1_3, self).__init__()
        self.convin = RBC(3, 64)
        self.rbc1 = RBC(64, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64,32)
        self.rbc1d = RBC(64,64)
        self.gf = GuidedFilter(r=2, eps=1e-2)

    
    def forward(self, x, IMGreys):
        _, _, img_shape, _ = x.size()
        hx = x
        imgreys = IMGreys
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)
        imgreys2 = F.upsample(imgreys, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        hx1h = self.gf(imgreys2, hx1h)
        _, _, img_shape, _ = imgreys2.size()

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)
        imgreys3 = F.upsample(imgreys2, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        hx2h = self.gf(imgreys3, hx2h)
        _, _, img_shape, _ = imgreys3.size()
        

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)
        imgreys4 = F.upsample(imgreys3, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        hx3h = self.gf(imgreys4, hx3h)
        _, _, img_shape, _ = imgreys4.size()

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)
        imgreys5 = F.upsample(imgreys4, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        hx4h = self.gf(imgreys5, hx4h)
        _, _, img_shape, _ = imgreys5.size()

        hx5 = self.rbc5(hx4h)
        hx5h = self.maxpool56(hx5)
        imgreys6 = F.upsample(imgreys5, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        hx5h = self.gf(imgreys6, hx5h)

        hx6 = self.rbc6(hx5h)
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx = self.gf(imgreys5, hx)

        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))
        hx = upsample(hx5d, hx4)
        hx = self.gf(imgreys4, hx)
        
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))
        hx = upsample(hx4d, hx3)
        hx = self.gf(imgreys3, hx)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)
        hx = self.gf(imgreys2, hx)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx = self.gf(imgreys, hx)

        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))
        
        decoder_outputs = {'6d': hx6d, '5d':hx5d, '4d':hx4d, '3d':hx3d, '2d':hx2d, 'output':hx1d+hxin}
        imgrey = {288:imgreys, 144:imgreys2, 72:imgreys3, 36:imgreys4, 18:imgreys5, 9:imgreys6}
        return decoder_outputs, imgrey
    
class Block9_3(nn.Module):
    def __init__(self):
        super(Block9_3, self).__init__()
        self.convin = RBC(64, 1)
        self.rbc1 = RBC(1, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64, 32)
        self.c1d = nn.Conv2d(64, 1, kernel_size = 3, padding=1)
        self.b1d = nn.BatchNorm2d(1)
        self.ReLu = nn.ReLU(True)
        self.gf = GuidedFilter(r=2, eps=1e-2)
    
    def forward(self, x, imgrey):
        imGrey = imgrey
        x_skip = x
        hx = x_skip['output']
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1) #144 hx1h
        hx1h = self.gf(imGrey[144], hx1h)



        hx2 = self.rbc2(hx1h + x_skip['2d'])   
        hx2h = self.maxpool23(hx2) #72
        hx2h = self.gf(imGrey[72], hx2h)

        hx3 = self.rbc3(hx2h+ x_skip['3d'])
        hx3h = self.maxpool34(hx3) #36
        hx3h = self.gf(imGrey[36], hx3h)

        hx4 = self.rbc4(hx3h+ x_skip['4d'])
        hx4h = self.maxpool45(hx4) #18
        hx4h = self.gf(imGrey[18], hx4h)

        hx5 = self.rbc5(hx4h+ x_skip['5d'])
        hx5h = self.maxpool56(hx5) #9
        hx5h = self.gf(imGrey[9], hx5h)

        hx6 = self.rbc6(hx5h+ x_skip['6d'])
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx = self.gf(imGrey[18], hx)
        
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))
        hx = upsample(hx5d, hx4)
        hx = self.gf(imGrey[36], hx)
        
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))
        hx = upsample(hx4d, hx3)
        hx = self.gf(imGrey[72], hx)
        
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)
        hx = self.gf(imGrey[144], hx)
        
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx = self.gf(imGrey[288], hx)
        
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))

        return hx1d_output
    


class UU3NET(nn.Module):
    def __init__(self):
        super(UU3NET, self).__init__()
        self.block1 = Block1_3()
        self.block9 = Block9_3()

    def forward(self, x, imgrey_in):
        hx = x
        hx1, imgrey_out = self.block1(hx, imgrey_in)
        hx2 = self.block9(hx1, imgrey_out)

        return F.sigmoid(hx2)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import pdb



class IMU_NN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 img_width,
                 img_height,
                 batch_size,
                 input_length,
                 num_heads,
                 depth,
                 **kwargs):
        super().__init__()


        self.conv_channels = 1
        self.img_width = img_width
        self.img_height = img_height
        self.d_model = out_channels
        self.batch_size = batch_size
        self.input_length = input_length
        self.in_channels = in_channels
        self.heads = num_heads
        self.depth = depth
        
        self.pos_embedding = PositionalEncoding(self.d_model, self.batch_size, self.img_height, self.img_width, self.input_length)
        self.Encoder = Encoder(self.d_model, self.heads, self.depth, self.img_height, self.img_width, self.input_length)
        #self.Encoder = Encoder()
        
        """
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=1,
                               kernel_size=3,
                               stride=1,
                               padding=(2)//2)
        self.bn = nn.BatchNorm2d(1)
        """
        self.generator = feature_generator(in_channels, self.conv_channels)
        
        self.fcn = nn.Linear(150, self.d_model)

    def forward(self, x):
        
        #out = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.01, inplace=False)
        #out = self.generator(x)
        #out = torch.flatten(out)
        #x = self.fcn(out)
        """
        b,n,h,w,l = x.shape
        out_list=[]
        feature_map = self.feature_embedding(x)
        enc_in = self.pos_embedding(feature_map)
        enc_out = self.Encoder(enc_in)
        """
        feature_map = self.feature_embedding(x).permute(4,0,1,2,3)
        enc_in = self.pos_embedding(feature_map)
        enc_out = self.Encoder(enc_in)
        
        return enc_out.permute(1,2,3,4,0).contiguous().view(self.batch_size,self.d_model,-1).permute(2,0,1)

    def feature_embedding(self,img):
        generator = feature_generator(self.in_channels, self.d_model)
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[:, :, :, :, i]))
        gen_img = torch.stack(gen_img, dim=-1)
        return gen_img




def imu_nn_baseline(in_channels, out_channels, img_width, img_height, batch_size, input_length, num_heads, depth):

    model = IMU_NN(
        in_channels,
        out_channels,
        img_width,
        img_height,
        batch_size,
        input_length,
        num_heads,
        depth
    )
    return model



#From

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        #pdb.set_trace()
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MultiConvAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.W0 = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.convs = [ConvAttention(self.d_model, int(self.d_model/self.heads)) for i in range(self.heads)]
        
    def forward(self, x, **kwargs):
        head_list = []
        for c in self.convs:
            head_list.append(c.forward(x.permute(1,2,3,4,0), dec=True))
        
        V_out = torch.cat(head_list, dim=1 )
        return torch.matmul(V_out.transpose(1,4), self.W0).transpose(1,4).permute(4,0,1,2,3)

class FeedForward(nn.Module):
    def __init__(self, input_length):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_length, input_length*4),
            nn.LeakyReLU(),
            nn.Linear(input_length*4, input_length),
        )
    def forward(self, x):
        return self.net(x.permute(1,2,3,4,0)).permute(4,0,1,2,3)
class ConvAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super(ConvAttention, self).__init__()
        
        self.d_model = d_model
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.d_model, out_channels=3*self.d_model, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=d_k, out_channels=d_k, kernel_size=5,padding=2)
        )
        
        self.Wiq = torch.nn.Parameter(torch.randn((self.d_model, d_k)))
        self.Wik = torch.nn.Parameter(torch.randn((self.d_model, d_k)))
        self.Wiv = torch.nn.Parameter(torch.randn((self.d_model, d_k)))
    def forward(self, x ,enc_out= None,dec=False):
        b,c,h,w,l = x.shape
        qkv_setlist = []
        Vout_list = []
        for i in range(l):
            qkv_setlist.append(self.conv1(x[...,i]))
        qkv_set = torch.stack(qkv_setlist,dim=-1)
        
        if dec:
            Q,K,V = torch.split(qkv_set,self.d_model,dim=1)
        else:
            Q,K,_ = torch.split(qkv_set,self.d_model,dim=1)
            V = enc_out
            
        Q = torch.matmul(Q.transpose(1,4), self.Wiq).transpose(1,4)
        K = torch.matmul(K.transpose(1,4), self.Wik).transpose(1,4)
        V = torch.matmul(V.transpose(1,4), self.Wiv).transpose(1,4)
        
        for i in range(l):
            Qi = rearrange(Q[...,i].unsqueeze(4).expand(-1,-1,-1,-1,l)+K, 'b n h w l -> (b l) n h w')
            tmp = rearrange(self.conv2(Qi),'(b l) n h w -> b n h w l',l=l)
            tmp = F.softmax(tmp, dim=4) #(b, n, h, w, l)
            tmp = tmp*V#np.multiply(tmp, torch.stack([V[i]]*l, dim=-1))
            Vout_list.append(torch.sum(tmp,dim=4)) #(b,n,h,w)
        
        Vout = torch.stack(Vout_list, dim=-1 )
        return Vout                            #(b,n,h,w,l)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, batch_size, img_height, img_width, input_length):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.d_model = d_model
        self.input_length = input_length
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table())
        

    def _get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):

            return_list = [torch.ones((self.batch_size,
                                       self.img_height,
                                       self.img_width))*(position / np.power(10000, 2 * (hid_j // 2) / self.d_model)) for hid_j in range(self.d_model)]
            return torch.stack(return_list, dim=1)
        sinusoid_table = np.array([np.array(get_position_angle_vec(pos_i)) for pos_i in range(self.input_length)])
        print(np.shape(sinusoid_table[0::2]))
        sinusoid_table[0::2] = np.sin(sinusoid_table[0::2])  # dim 2i
        sinusoid_table[1::2] = np.cos(sinusoid_table[1::2])  # dim 2i+1
        print(np.shape(sinusoid_table))

        return torch.from_numpy(sinusoid_table)#torch.stack(sinusoid_table, dim=-1)

    def forward(self, x):
        '''

        :param x: (b, channel, h, w, seqlen)
        :return:
        '''
        return x + self.pos_table.clone().detach()

class Encoder(nn.Module):
    def __init__(self, d_model, heads, depth, img_height, img_width, input_length):
        super().__init__()
        self.layers = nn.ModuleList([])
        

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([d_model,img_height,img_width],
                                 MultiConvAttention(d_model, heads))),
                Residual(PreNorm([d_model,img_height,img_width],
                                 FeedForward(input_length)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #pdb.set_trace()
            x = ff(x)
        return x



class feature_generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(feature_generator, self).__init__()
        self.num_hidden = [4,6,8]
        self.filter_size = 3
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.num_hidden[0],
                               kernel_size=self.filter_size,
                               stride=1,
                               padding=(self.filter_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels=self.num_hidden[0],
                               out_channels=self.num_hidden[1],
                               kernel_size=self.filter_size,
                               stride=1,
                               padding=(self.filter_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=self.num_hidden[1],
                               out_channels=self.num_hidden[2],
                               kernel_size=self.filter_size,
                               stride=1,
                               padding=(self.filter_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=self.num_hidden[2],
                               out_channels=out_channels,
                               kernel_size=self.filter_size,
                               stride=1,
                               padding=(self.filter_size-1)//2)
        self.bn1 = nn.BatchNorm2d(self.num_hidden[0])
        self.bn2 = nn.BatchNorm2d(self.num_hidden[1])
        self.bn3 = nn.BatchNorm2d(self.num_hidden[2])
        self.bn4 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope=0.01, inplace=False)
        return out



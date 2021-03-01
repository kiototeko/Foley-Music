import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



class IMU_NN(nn.Module):

    def __init__(self,
                 in_channels,
                 num_class,
                 **kwargs):
        super().__init__()


        self.conv_model = 32
        self.img_width = 3
        self.d_model = num_class
        self.batch_size = 3
        self.input_length = 2
        
        #self.pos_embedding = PositionalEncoding()
        #self.Encoder = Encoder(dim, depth, heads, mlp_dim, dropout)
        #self.Encoder = Encoder()
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=1,
                               kernel_size=3,
                               stride=1,
                               padding=(2)//2)
        self.bn = nn.BatchNorm2d(1)

        self.fcn = nn.Linear(150, self.d_model)

    def forward(self, x):
        #pdb.set_trace()
        out = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.01, inplace=False)
        #out = torch.flatten(out)
        x = self.fcn(out)
        """
        b,n,h,w,l = x.shape
        out_list=[]
        feature_map = self.feature_embedding(x)
        enc_in = self.pos_embedding(feature_map)
        enc_out = self.Encoder(enc_in)
        """
        return x

    def feature_embedding(self,img):
        generator = feature_generator()
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[:, :, :, :, i]))
        gen_img = torch.stack(gen_img, dim=-1)
        return gen_img




def imu_nn_baseline(in_channels: int, out_channels: int):

    model = IMU_NN(
        in_channels,
        out_channels
    )
    return model



#From

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(self.input_length, self.input_length*4),
            nn.LeakyReLU(),
            nn.Linear(self.input_length*4, self.input_length),
        )
    def forward(self, x):
        return self.net(x)
class ConvAttention(nn.Module):
    def __init__(self):
        super(ConvAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel=self.conv_model, out_channel=3*self.conv_model, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel=self.conv_model, out_channel=self.conv_model, kernel_size=5,padding=2)
        )
    def forward(self, x ,enc_out= None,dec=False):
        b,c,h,w,l = x.shape
        qkv_setlist = []
        Vout_list = []
        for i in l:
            qkv_setlist.append(self.conv1(x[...,i]))
        qkv_set = torch.stack(qkv_setlist,dim=-1)
        if dec:
            Q,K,V = torch.split(qkv_set,self.conv_model,dim=1)
        else:
            Q,K,_ = torch.split(qkv_set,self.conv_model,dim=1)
            V = enc_out

        for i in l:
            Qi = rearrange([Q[...,i]]*l+K, 'b n h w l -> (b l) n h w')
            tmp = rearrange(self.conv2(Qi),'(b l) n h w -> b n h w l',l=l)
            tmp = F.softmax(tmp, dim=4) #(b, n, h, w, l)
            tmp = np.multiply(tmp, torch.stack([V[i]]*l, dim=-1))
            Vout_list.append(torch.sum(tmp,dim=4)) #(b,n,h,w)
        Vout = torch.stack(Vout_list, dim=-1 )
        return Vout                            #(b,n,h,w,l)

class PositionalEncoding(nn.Module):

    def __init__(self):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table())

    def _get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):

            return_list = [torch.ones((self.batch_size,
                                       self.img_width,
                                       self.img_width))*(position / np.power(10000, 2 * (hid_j // 2) / self.conv_model)) for hid_j in range(self.conv_model)]
            return torch.stack(return_list, dim=1)
        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(self.input_length)]
        sinusoid_table[0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.stack(sinusoid_table, dim=-1)

    def forward(self, x):
        '''

        :param x: (b, channel, h, w, seqlen)
        :return:
        '''
        return x + self.pos_table.clone().detach()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.conv_model,self.img_width,self.img_width],
                                 ConvAttention())),
                Residual(PreNorm([self.conv_model,self.img_width,self.img_width],
                                 FeedForward()))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x



class feature_generator(nn.Module):
    def __init__(self):
        super(feature_generator, self).__init__()
        self.num_hidden = [9,8,7]
        self.filter_size = 3
        self.conv1 = nn.Conv2d(in_channels=3,
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
                               out_channels=self.conv_model,
                               kernel_size=self.filter_size,
                               stride=1,
                               padding=(self.filter_size-1)//2)
        self.bn1 = nn.BatchNorm2d(self.num_hidden[0])
        self.bn2 = nn.BatchNorm2d(self.num_hidden[1])
        self.bn3 = nn.BatchNorm2d(self.num_hidden[2])
        self.bn4 = nn.BatchNorm2d(self.conv_model)
        
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope=0.01, inplace=False)
        return out



    

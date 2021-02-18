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


        self.fcn = nn.Linear(in_channels, num_class)

    def forward(self, x):
        #pdb.set_trace()
       

        # prediction
        x = self.fcn(x)
        
        return x




def imu_nn_baseline(in_channels: int, out_channels: int):

    model = IMU_NN(
        in_channels,
        out_channels
    )
    return model


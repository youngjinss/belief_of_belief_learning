import torch
from torch import nn
from ToMnetEngine.PredNet import PredNet
from ToMnetEngine.CharNet import CharNet

"""
@Author Filip Borowiak 
"""


class ToMnet(nn.Module):
 def __init__(self, Batch: int, ResidualBlocks: int,  N_echar: int, out_channels: int, 
              Max_trajectory_size: int, Width: int, Height: int, Depth: int):
    super(ToMnet, self).__init__()

    self.ts = Max_trajectory_size
    self.W = Width
    self.H = Height
    self.C = Depth
    self.B = Batch
    self.resN = ResidualBlocks
    self.Length_E = N_echar
    self.out_channels = out_channels


    self.char_net = CharNet(Batch=self.B, 
                            ResidualBlocks=self.resN,
                            N_echar=self.Length_E,
                            channels_in = self.C,
                            out_channels=self.out_channels,
                            time_frame=self.ts)
    
    self.pred_net = PredNet(Batch=self.B,
                            ResidualBlocks=self.resN,
                            E_char=self.Length_E,
                            out_channels=self.out_channels,
                            time_frame=self.ts)


 def SaveModel(self, destination):
   torch.save(self.state_dict(), destination)
   
   
 def forward(self, data):
    input_trajectory = data[0]   # input_traj         
    input_current_state = data[1] #  input_current    

    e_char = self.char_net(input_trajectory)

    e_char_new = torch.concat([e_char, e_char], dim=1)
    e_char_new = e_char_new[..., 0:13]

    e_char_new = torch.unsqueeze(e_char_new, dim=-1)

    e_char_new = torch.repeat_interleave(e_char_new, repeats=13, dim=-1)

    e_char_new = torch.unsqueeze(e_char_new, dim=1)
    


    mixed_data = torch.cat((e_char_new,input_current_state), dim=1)

    pred = self.pred_net(mixed_data)
    
    return pred
 


   
   
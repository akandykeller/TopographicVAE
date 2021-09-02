from torch import nn

def MLP_Encoder(s_dim, n_cin, n_hw):
    model = nn.Sequential(
                nn.Conv2d(n_cin, s_dim*3,
                    kernel_size=n_hw, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(s_dim*3, s_dim*2,
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(s_dim*2, s_dim*2,
                    kernel_size=1, stride=1, padding=0))
    return model

def MLP_Decoder(s_dim, n_cout, n_hw):
    model = nn.Sequential(
                nn.ConvTranspose2d(s_dim, s_dim*2, 
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(s_dim*2, s_dim*3, 
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(s_dim*3, n_cout, 
                    kernel_size=n_hw, stride=1, padding=0),
                nn.Sigmoid())
    return model
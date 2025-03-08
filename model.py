import torch
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=1):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        # self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)

        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, x):
        x = self.leakyrelu(self.dense1(x))
        # x = self.leakyrelu(self.dense2(x))
        x = self.dense3(x)

        return x

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


class Decoder_Res(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder_Res, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(hdim, cc * 4 * 4),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


class Project(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Project, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        return self.projector(x)


class Encoder_CADVAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, feat_dim=32, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder_CADVAE, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.fc = nn.Linear((cc) * 4 * 4, hdim)

        self.proj_x = Project(hdim, 2 * (hdim - 3 * feat_dim))
        self.proj_y = Project(hdim, feat_dim * 2)
        self.proj_r = Project(hdim, feat_dim * 2)
        self.proj_s = Project(hdim, feat_dim * 2)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)

        #         mu, logvar = y.chunk(2, dim=1)
        mu_x, logvar_x = self.proj_x(y).chunk(2, dim=1)
        mu_y, logvar_y = self.proj_y(y).chunk(2, dim=1)
        mu_r, logvar_r = self.proj_r(y).chunk(2, dim=1)
        mu_s, logvar_s = self.proj_s(y).chunk(2, dim=1)

        z_x = self.reparameterize(mu_x, logvar_x)
        z_y = self.reparameterize(mu_y, logvar_y)
        z_r = self.reparameterize(mu_r, logvar_r)
        z_s = self.reparameterize(mu_s, logvar_s)

        return (z_x, z_y, z_s, z_r), (mu_x, mu_y, mu_s, mu_r), (logvar_x, logvar_y, logvar_s, logvar_r)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)


class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        #         output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output

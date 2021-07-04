import torch
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from typing import Union

class Waifu2x(nn.Sequential):
    def __init__(self):
        layers = [
             nn.ZeroPad2d(7),
             nn.Conv2d(3, 16, 3, 1, 0),
             nn.LeakyReLU(0.1, inplace=True),
             nn.Conv2d(16, 32, 3, 1, 0),
             nn.LeakyReLU(0.1, inplace=True),
             nn.Conv2d(32, 64, 3, 1, 0),
             nn.LeakyReLU(0.1, inplace=True),
             nn.Conv2d(64, 128, 3, 1, 0),
             nn.LeakyReLU(0.1, inplace=True),
             nn.Conv2d(128, 128, 3, 1, 0),
             nn.LeakyReLU(0.1, inplace=True),
             nn.Conv2d(128, 256, 3, 1, 0),
             nn.LeakyReLU(0.1, inplace=True),
             nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False)
         ]
        super(Waifu2x, self).__init__(*layers)

        # weights ported from https://github.com/nagadomi/waifu2x
        ckpt_url = "https://onedrive.live.com/download?cid=467C8AA2DE5C1D02&resid=467C8AA2DE5C1D02%21155&authkey=AFLQwtj_9nVIYy8"
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url, file_name="upconv7_noise2_scale2.0x.pt", progress=False)
        self.load_state_dict(ckpt)
        
    def forward(self, img: Union[torch.Tensor, Image.Image]):
        if isinstance(img, Image.Image):
            return self.forward_pil(img)
        return super().forward(img)
    
    def forward_pil(self, img: Image.Image):
        device = self[1].weight.device
        img = to_tensor(img).unsqueeze(0).to(device)
        out = self(img)[0].clip(0, 1)
        return to_pil_image(out)

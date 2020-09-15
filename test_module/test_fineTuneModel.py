
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
print(sys.path)

from models import build_model
import torch
from utils.util import *

if __name__ == "__main__":

    model = build_model('FineTuneStylegan', image_size=64)
    print(model.G)
    inp1 = torch.randn(3, 5, 512)
    noise = custom_image_nosie(3, 100)
    inp2 = latent_to_nosie(model.N, noise)
    fake_img=model.G(inp1,inp2)
    
    print(model.G(inp1,inp2).shape)
    print("complete check G")

    # print(model.D(fake_img))
    # print("complete check D")

    print(model.E(fake_img).shape)
    print(model.E(fake_img))
    print("complete check E")

    print('complete check.')

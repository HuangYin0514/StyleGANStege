
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
print(sys.path)
from models import build_model
import torch

if __name__ == "__main__":

    model = build_model('ExtractNetSimilarE', image_size=64)
    print(model)
    inp = torch.randn(3,3,64,64)
    print(model(inp).shape)

    print('complete check.')

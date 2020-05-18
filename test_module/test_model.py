
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
print(sys.path)
from models import build_model

if __name__ == "__main__":

    model = build_model('StyleGAN2', image_size=64)
    print(model)

    print('complete check.')

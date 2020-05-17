
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
print(sys.path)
from dataloader import getDataLoader, check_data

if __name__ == "__main__":

    # dataset
    loader = getDataLoader('Celeba', 64, '/home/hy/vscode/reid-custom/data/Market-1501-v15.09.15')

    data = next(loader)
    print(data.shape)
    check_data(data, "./dataloader/checkdata.jpg")

    print('complete check.')

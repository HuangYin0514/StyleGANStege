
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
print(sys.path)
from dataloader import getDataLoader, check_data

if __name__ == "__main__":

    # dataset
    loader = getDataLoader(dataset='Celeba', batch_size=3,dataset_path= '/home/hy/vscode/StyleGANStege/data/celeba')

    data = next(loader)
    print(data.shape)
    check_data(data, "./dataloader/checkdata.jpg")

    print('complete check.')

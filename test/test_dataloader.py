
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
print(sys.path)
from dataloader import getDataLoader,check_data

if __name__ == "__main__":

    # dataset
    loader = 

    data = next(iter(loader))
    check_data(data, "./dataloader/checkdata.jpg")

    print('complete check.')

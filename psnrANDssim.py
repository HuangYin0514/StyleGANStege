from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import cv2
import os


def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    psnr = 0.0
    ssim = 0.0
    n = 0
    path = 'experiments/sp/dn/'
    sp_dir = os.listdir(path)
    # -------------------------------------------
    for image1 in sp_dir:
        for image2 in sp_dir:
            if image1 == image2:
                continue

            img1 = cv2.imread(path+image1)
            img2 = cv2.imread(path+image2)
            img1_grey = to_grey(img1)
            img2_grey = to_grey(img2)

            res_psnr = compare_psnr(img1_grey, img2_grey)
            psnr += res_psnr
            res_ssim = compare_ssim(img1_grey, img2_grey)
            ssim += res_ssim

            n += 1
    # -------------------------------------------
    psnr = psnr / n
    ssim = ssim / n
    print('n of number is ', n)
    print("average psnr = ", psnr)
    print("average ssim = ", ssim)

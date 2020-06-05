from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import cv2


def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    img1 = cv2.imread('experiments/sp/1.jpg')
    img2 = cv2.imread('experiments/sp/2.jpg')
    img1_grey = to_grey(img1)
    img2_grey = to_grey(img2)

    res_psnr = compare_psnr(img1_grey, img2_grey)
    print(res_psnr)
    
    res_ssim = compare_ssim(img1_grey, img2_grey)
    print(res_ssim)

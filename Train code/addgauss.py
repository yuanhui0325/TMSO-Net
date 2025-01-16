import os
import cv2
import numpy as np

def get_file_list(root_dir):
    file_list = []
    flist = os.listdir(root_dir)
    for i in range(0,len(flist)):
        path = os.path.join(root_dir,flist[i])
        file_list.append(path)
    return file_list

def add_gaussian_noise(img,param=0.03,grayscale=256):
    w = img.shape[1]
    h = img.shape[0]
    newimg = np.zeros((h, w, 3), np.uint8)
    for x in range(0, h):
        for y in range(0, w, 2):
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
            z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

            fxy_0 = int(img[x, y, 0] + z1)
            fxy1_0 = int(img[x, y + 1, 0] + z2)
            fxy_1 = int(img[x, y, 1] + z1)
            fxy1_1 = int(img[x, y + 1, 1] + z2)
            fxy_2 = int(img[x, y, 2] + z1)
            fxy1_2 = int(img[x, y + 1, 2] + z2)

            # f(x,y,z)
            if fxy_0 < 0:
                fxy_val_0 = 0
            elif fxy_0 > grayscale - 1:
                fxy_val_0 = grayscale - 1
            else:
                fxy_val_0 = fxy_0

            if fxy_1 < 0:
                fxy_val_1 = 0
            elif fxy_1 > grayscale - 1:
                fxy_val_1 = grayscale - 1
            else:
                fxy_val_1 = fxy_1

            if fxy_2 < 0:
                fxy_val_2 = 0
            elif fxy_2 > grayscale - 1:
                fxy_val_2 = grayscale - 1
            else:
                fxy_val_2 = fxy_2

            # f(x,y+1,z)
            if fxy1_0 < 0:
                fxy1_val_0 = 0
            elif fxy1_0 > grayscale - 1:
                fxy1_val_0 = grayscale - 1
            else:
                fxy1_val_0 = fxy1_0

            if fxy1_1 < 0:
                fxy1_val_1 = 0
            elif fxy1_1 > grayscale - 1:
                fxy1_val_1 = grayscale - 1
            else:
                fxy1_val_1 = fxy1_1

            if fxy1_2 < 0:
                fxy1_val_2 = 0
            elif fxy1_2 > grayscale - 1:
                fxy1_val_2 = grayscale - 1
            else:
                fxy1_val_2 = fxy1_2

            newimg[x, y, 0] = fxy_val_0
            newimg[x, y, 1] = fxy_val_1
            newimg[x, y, 2] = fxy_val_2
            newimg[x, y + 1, 0] = fxy1_val_0
            newimg[x, y + 1, 1] = fxy1_val_1
            newimg[x, y + 1, 2] = fxy1_val_2
    return newimg

root_dir = 'E:\epinet-master\hci_dataset\additional\'
new_root_dir = 'E:\epinet-master\hci_dataset\additional_guass\'
fnm_list = get_file_list(root_dir)
for fnm in fnm_list:
    img = cv2.imread(fnm)
    newimg = add_gaussian_noise(img,param=0.03,grayscale=256)
    newImgName = os.path.basename(fnm)
    newImgName = os.path.join(new_root_dir,newImgName)
    print(newImgName)
    cv2.imwrite(newImgName, newimg)
    cv2.waitKey()
    cv2.destroyAllWindows()







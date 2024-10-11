from modules.dataset_tools import mhio
import matplotlib.pyplot as plt
from modules.dataset_tools.image_tools import plot_coords
import cv2

global_location = mhio.load_json("modules/dataset_tools/metadata/aligned_Global_locations.txt")
true = global_location['foil_1_2_aligned']


def display(display_list, title_list=None):
    if title_list is None:
        title_list = ['Dirty Image', 'Clean Image', 'Predicted Image']
    plt.figure(figsize=(30, 30))
    title = title_list
    for count in range(len(display_list)):
        plt.subplot(1, len(display_list), count + 1)
        plt.title(title[count])
        plt.imshow(display_list[count], cmap="gray")
        plt.axis('off')
    plt.show()


def display_fn(display_list, false_neg_dr_ixy):
    for x in range(len(false_neg_dr_ixy)):
        plt.figure(figsize=(30, 30))
        title = ['Dirty Image', 'Clean Image', 'Predicted Image']
        plt.subplot(1, len(display_list), 1)
        plt.title(title[0])
        plt.imshow(display_list[0][false_neg_dr_ixy[x][0] + 1, :, :, 8], cmap="gray")
        plt.axis('off')
        plt.subplot(1, len(display_list), 2)
        plt.title(title[1])
        plt.imshow(display_list[1][false_neg_dr_ixy[x][0] + 1, :, :], cmap="gray")
        plt.axis('off')
        plt.subplot(1, len(display_list), 3)
        plt.title(title[2])
        plt.imshow(display_list[2][false_neg_dr_ixy[x][0] + 1, :, :], cmap='gray')
        plt.plot(false_neg_dr_ixy[x][1], false_neg_dr_ixy[x][2], "gx")
        plt.axis('off')
        plt.show()


def binary(array, thresh):
    for i in range(array.shape[0]):
        array[i, :, :] = cv2.threshold(array[i, :, :], thresh, 255, cv2.THRESH_BINARY)[1]
    return array


def extract_coord(lst, id):
    if id == "x":
        return [item[0] for item in lst]
    if id == "y":
        return [item[1] for item in lst]


def test_overlap(true, pred, tolerance=10):  # copied from prev pit finder code
    """
    Calculate absoloute true positive / negative rates per image pair assuming XY = true
    """
    xnew, ynew = [], []
    for x, y in zip(extract_coord(true, "x"), extract_coord(true, "y")):
        for x2, y2 in zip(extract_coord(pred, "x"), extract_coord(pred, "y")):
            if (x > (x2 - tolerance)) & (x < (x2 + tolerance)) & (y > (y2 - tolerance)) & (y < (y2 + tolerance)):
                xnew.append(x), ynew.append(y)
    true_pos = len(xnew)
    true_neg = len(extract_coord(true, "x")) - len(xnew)  # No. true examples missed
    false_pos = len(extract_coord(pred, "x")) - len(xnew)
    return true_pos, true_neg, false_pos


def false_pos(false_pos_ind, predicted_list):
    print("False Positive list")
    for i in range(len(false_pos_ind[0])):
        print(predicted_list[int(false_pos_ind[0][i])])


def false_neg(false_neg_ind, true_global=true):
    print("False Negative list")
    for i in range(len(false_neg_ind[0])):
        print(true_global[int(false_neg_ind[0][i])])

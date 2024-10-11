import numpy as np
from modules.dataset_tools import coordinates, mhio
import matplotlib.pyplot as plt
from modules.compare.tools import extract_coord


def plot(list, dist, pred_list, name="FCN_front", log=False, save=False):
    """
    Plots spatial resolution of pits in terms of dx, dy, and ds.

    Parameters:
    - list: List of lists with first index as dx and second as dy.
    - dist: List containing the nearest distance of each predicted pit.
    - pred_list: List containing the predicted pits.
    - name: Name used in saving and title of the plot.
    - log: Switch log on for all prediction lists.
    - save: Whether to save the plots or not (default is False).

    Returns:
    - None
    """
    bins = 100
    plt.figure(figsize=(15, 15), dpi=100)
    if log:
        hist, bins = np.histogram(extract_coord(list, "x"), bins=bins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(extract_coord(list, "x"), bins=logbins)
        plt.xscale('log')
    else:
        plt.hist(extract_coord(list, "x"), bins=20)
    plt.title("Spatial resolution - dx\n" + name)
    plt.xlabel("dx")
    plt.ylabel("count")
    if save:
        plt.savefig("plots/" + name + "_dx", dpi=100)
    plt.show()

    # Plot dy
    plt.figure(figsize=(15, 15), dpi=100)
    if log:
        hist, bins = np.histogram(extract_coord(list, "y"), bins=bins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(extract_coord(list, "y"), bins=logbins)
        plt.xscale('log')
    else:
        plt.hist(extract_coord(list, "y"), bins=20)
    plt.title("Spatial resolution - dy\n" + name)
    plt.xlabel("dy")
    plt.ylabel("count")
    if save:
        plt.savefig("plots/" + name + "_dy", dpi=100)
    plt.show()

    # Plot ds (distance)
    plt.figure(figsize=(15, 15), dpi=100)
    if log:
        hist, bins = np.histogram(dist, bins=bins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(dist, bins=logbins)
        plt.xscale('log')
    else:
        plt.hist(dist, bins=20)
    plt.title("Spatial resolution - ds\n" + name)
    plt.xlabel("ds")
    plt.ylabel("count")
    if save:
        plt.savefig("plots/" + name + "_ds", dpi=100)
    plt.show()

    # Scatter plot dx vs dy
    plt.figure(figsize=(15, 15), dpi=100)
    plt.scatter(extract_coord(list, "x"), extract_coord(list, "y"))
    plt.title("Spatial resolution - dx vs dy\n" + name)
    plt.xlabel("dx")
    plt.ylabel("dy")
    if save:
        plt.savefig("plots/" + name + "_dx_dy", dpi=100)
    plt.show()

    # Scatter plot x vs dx
    plt.figure(figsize=(15, 15), dpi=100)
    plt.scatter(extract_coord(pred_list, "x"), extract_coord(list, "x"))
    plt.title("Spatial resolution - x vs dx\n" + name)
    plt.xlabel("x")
    plt.ylabel("dx")
    if save:
        plt.savefig("plots/" + name + "_x_dx", dpi=100)
    plt.show()

    # Scatter plot y vs dy
    plt.figure(figsize=(15, 15), dpi=100)
    plt.scatter(extract_coord(pred_list, "y"), extract_coord(list, "y"))
    plt.title("Spatial resolution - y vs dy\n" + name)
    plt.xlabel("y")
    plt.ylabel("dy")
    if save:
        plt.savefig("plots/" + name + "_y_dy", dpi=100)
    plt.show()


def get_spacial_reso_FCN_front():
    """
    Computes the spatial resolution for FCN front pits, comparing the predicted and true pits.

    Returns:
    - None
    """
    true_eval = mhio.load_json("Aligned_clean.txt")  # True pits in Lewis's and Kanik's data
    CS = coordinates.Coord_system()

    predicted_list = mhio.load_json("predicted_etch_pits/FCN/final_pr.txt")
    predicted_list_eval = []

    # Filter invalid pits from the predicted list
    for i in range(len(predicted_list)):
        tmp = CS['c1'].get_ixy(predicted_list[i])
        if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
            continue
        tmp = CS.convert(tmp, 'cd')
        if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
            continue
        predicted_list_eval.append(predicted_list[i][:])

    # Compare predicted pits with true pits using a tolerance of 15
    x = coordinates.Overlap(np.array(predicted_list_eval), np.array(true_eval), tolerance=15)
    coordinates.tally(x)

    true_pred_ind = x.A_in_B()
    true_pred = [predicted_list_eval[int(true_pred_ind[0][i])] for i in range(len(true_pred_ind[0]))]

    # Calculate the nearest distances for all predicted pits
    nearest_point_ind_of_all_pred = x.nearest_a
    nearest_point_in_true_of_all_pred = []
    nearest_x_y_dist_of_all_pred = []
    for i in range(len(nearest_point_ind_of_all_pred)):
        nearest_point_in_true_of_all_pred.append(true_eval[int(nearest_point_ind_of_all_pred[i])])
        nearest_x_y_dist_of_all_pred.append([abs(nearest_point_in_true_of_all_pred[i][0] - predicted_list_eval[i][0]),
                                             abs(nearest_point_in_true_of_all_pred[i][1] - predicted_list_eval[i][1])])

    nearest_distance_of_all_pred = x.closest_dist_a

    # Plot spatial resolution for all predictions
    plot(nearest_x_y_dist_of_all_pred, nearest_distance_of_all_pred, nearest_point_in_true_of_all_pred,
         name="FCN_front_all_pred", log=True, save=True)

    # Now for the true predictions
    y = coordinates.Overlap(np.array(true_pred), np.array(true_eval), tolerance=15)
    nearest_point_ind_of_true_pred = y.nearest_a
    nearest_point_in_true_eval_of_true_pred = []
    nearest_x_y_dist_of_true_pred = []
    for i in range(len(nearest_point_ind_of_true_pred)):
        nearest_point_in_true_eval_of_true_pred.append(true_eval[int(nearest_point_ind_of_true_pred[i])])
        nearest_x_y_dist_of_true_pred.append([abs(nearest_point_in_true_eval_of_true_pred[i][0] - true_pred[i][0]),
                                              abs(nearest_point_in_true_eval_of_true_pred[i][1] - true_pred[i][1])])

    nearest_distance_of_true_pred = y.closest_dist_a

    # Plot spatial resolution for true predictions
    plot(nearest_x_y_dist_of_true_pred, nearest_distance_of_true_pred, nearest_point_in_true_eval_of_true_pred,
         name="FCN_front_true_pred", save=True)


# Call the main function to evaluate spatial resolution
if __name__ == "__main__":
    get_spacial_reso_FCN_front()

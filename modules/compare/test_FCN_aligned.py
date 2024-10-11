import numpy as np
from modules.dataset_tools import coordinates, mhio
import matplotlib.pyplot as plt
from modules.compare.tools import extract_coord


def plot(true, predicted, name, save=False):
    """
    Plots the comparison between true and predicted pits.

    Parameters:
    - true: List of true pits.
    - predicted: List of predicted pits.
    - name: Name for the plot file.
    - save: Boolean to decide if the plot should be saved as a file.

    Returns:
    - None
    """
    plt.figure(figsize=(15, 15), dpi=100)
    tr = plt.scatter(x=extract_coord(true, "x"), y=extract_coord(true, "y"), marker="o", s=30, color="green")
    pr = plt.scatter(x=extract_coord(predicted, "x"), y=extract_coord(predicted, "y"), marker="x", s=30, color="blue")
    plt.title("Comparing Predicted Pits\nFCN.py")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend((tr, pr), ('True pits', 'Predicted pits_FCN'), scatterpoints=1, loc=0, ncol=2, fontsize=8)
    if save:
        plt.savefig("plots/" + name, dpi=100)
    plt.show()


def evaluate_FCN_aligned(tolerance=15):
    """
    Compares predicted pits from FCN model (aligned data) with true pits, identifies false negatives,
    false positives, true positives, and fragments, and saves the results.

    Parameters:
    - tolerance: The tolerance value for comparing true and predicted pits.

    Returns:
    - None
    """

    # Load predicted pits (aligned) and true pits (aligned clean)
    predicted_list = mhio.load_json("predicted_etch_pits/FCN/aligned.txt")
    C1_true_eval = mhio.load_json("Aligned_clean.txt")  # True pits from Lewis's and Kanik's data
    CS = coordinates.Coord_system()

    predicted_list_eval = []

    # Filter out invalid pits from the predicted list
    for i in range(len(predicted_list)):
        tmp = CS['c1'].get_ixy(predicted_list[i])
        if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
            continue
        tmp = CS.convert(tmp, 'cd')  # Convert to cd coordinates
        if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
            continue
        predicted_list_eval.append(predicted_list[i][:])

    # Output the count of true and predicted pits
    print("# of true pits: {}".format(len(C1_true_eval)))
    print("# of predicted pits: {}".format(len(predicted_list_eval)))
    print("Comparing predicted pits front with true pits...")

    # Compare the true and predicted pits using a tolerance value
    x = coordinates.Overlap(np.array(predicted_list_eval), np.array(C1_true_eval), tolerance=tolerance)
    coordinates.tally(x)

    # Extract indices for false negatives, false positives, and true positives
    false_neg_ind = x.B_not_A()
    false_pos_ind = x.A_not_B()
    true_pos_ind = x.B_in_A()
    true_pred_ind = x.A_in_B()

    # Initialize lists for different categories of pits
    false_neg, false_pos, true_pos, true_pred = [], [], [], []

    # Populate the lists based on the indices
    for i in range(len(false_neg_ind[0])):
        false_neg.append(C1_true_eval[int(false_neg_ind[0][i])])
    for i in range(len(false_pos_ind[0])):
        false_pos.append(predicted_list_eval[int(false_pos_ind[0][i])])
    for i in range(len(true_pos_ind[0])):
        true_pos.append(C1_true_eval[int(true_pos_ind[0][i])])
    for i in range(len(true_pred_ind[0])):
        true_pred.append(predicted_list_eval[int(true_pred_ind[0][i])])

    # Output the number of false negatives and false positives
    print("False Neg = {}".format(len(false_neg)))
    print("False Pos = {}".format(len(false_pos)))

    # Load global true pit data and filter out invalid pits
    global_location = mhio.load_json("modules/dataset_tools/metadata/all_aligned_Global_locations.txt")
    C1_true = global_location['c1']
    C1_true_eval = []
    for i in range(len(C1_true)):
        tmp = CS['c1'].get_ixy(C1_true[i])
        if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
            continue
        C1_true_eval.append(C1_true[i][:])

    # Identify fragments (false positives that overlap with true pits)
    print("Identifying fragments (false positives that overlap with true pits)...")
    z = coordinates.Overlap(np.array(false_pos), np.array(C1_true_eval), tolerance=tolerance)
    coordinates.tally(z)
    frag_ind = z.A_in_B()
    frag = [false_pos[int(frag_ind[0][i])] for i in range(len(frag_ind[0]))]

    # Save false positives, false negatives, true positives, and fragments to JSON files
    mhio.save_json(false_pos, "predicted_list/FCN/aligned/final_false_pos.txt")
    mhio.save_json(false_neg, "predicted_list/FCN/aligned/final_false_neg.txt")
    mhio.save_json(true_pos, "predicted_list/FCN/aligned/final_true_pos.txt")
    mhio.save_json(frag, "predicted_list/FCN/aligned/final_fragments.txt")
    mhio.save_json(predicted_list_eval, "predicted_list/FCN/aligned/final_pred_list.txt")

    # Plot the comparison of true vs predicted pits
    plot(C1_true_eval, predicted_list_eval, "compare_aligned_FCN", True)


# If the script is executed as a standalone program, run the evaluation
if __name__ == "__main__":
    evaluate_FCN_aligned()

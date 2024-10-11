import numpy as np
from modules.dataset_tools import coordinates, mhio
import matplotlib.pyplot as plt
from modules.compare.tools import extract_coord

'''
Compare pits detected by two different methods (front and back) and identify common, only front, and only back pits.
'''


def plot(list1, list2, name1, name2, save=False):
    """
    Plots two sets of pits (from different methods) and compares them.

    Parameters:
    - list1, list2: Lists containing pit coordinates from two methods.
    - name1, name2: Labels for the two sets of pits to be displayed in the plot legend.
    - save: Whether to save the plot to a file (default is False).

    Returns:
    - None
    """
    plt.figure(figsize=(15, 15), dpi=100)
    C1 = plt.scatter(x=extract_coord(list1, "x"), y=extract_coord(list1, "y"), marker="o", s=30, color="green")
    Cr = plt.scatter(x=extract_coord(list2, "x"), y=extract_coord(list2, "y"), marker="x", s=30, color="blue")
    plt.title(f"{name1} and {name2}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend((C1, Cr), (name1, name2), scatterpoints=1, loc=0, ncol=2, fontsize=8)
    if save:
        plt.savefig(f"plots/CNN_pr_alignment_{name1}_and_{name2}.png", dpi=100)
    plt.show()


def compare_CNN_aligned(tolerance=12):
    """
    Compares pits detected by the CNN model from front and back exposures, identifying common, only-front, and only-back pits.
    Saves the common pits and plots the comparison.

    Parameters:
    - tolerance: The tolerance value for considering two pits as overlapping.

    Returns:
    - None
    """

    # Load front and back evaluation data
    front_eval = mhio.load_json("predicted_etch_pits/CNN/final_pr.txt")
    back_eval = mhio.load_json("predicted_etch_pits/CNN/final_reverse_pr.txt")

    # Output basic information about the pits
    print("# of front pits: {}".format(len(front_eval)))
    print("# of back pits: {}".format(len(back_eval)))

    # Compare front and back pits using a tolerance value
    x = coordinates.Overlap(np.array(front_eval), np.array(back_eval), tolerance=tolerance)
    coordinates.tally(x)

    # Extract indices for only-back, only-front, and common pits
    only_back_ind = x.B_not_A()
    only_front_ind = x.A_not_B()
    common_ind = x.A_in_B()

    # Initialize lists for only-back, only-front, and common pits
    only_back = []  # Pits in back exposure but not in front
    only_front = []  # Pits in front exposure but not in back
    common = []  # Common pits in both exposures

    # Populate the lists based on indices
    for i in range(len(only_back_ind[0])):
        only_back.append(back_eval[int(only_back_ind[0][i])])
    for i in range(len(only_front_ind[0])):
        only_front.append(front_eval[int(only_front_ind[0][i])])
    for i in range(len(common_ind[0])):
        common.append(front_eval[int(common_ind[0][i])])

    # Output the count of only-back, only-front, and common pits
    print("Only back = {}".format(len(only_back)))
    print("Only front = {}".format(len(only_front)))
    print("Common = {}".format(len(common)))

    # Save the common pits to a JSON file
    mhio.save_json(only_front, "predicted_list/CNN/Aligned_Front_Not_back.txt")
    mhio.save_json(only_back, "predicted_list/CNN/Aligned_Back_Not_Front.txt")
    mhio.save_json(common, "predicted_etch_pits/CNN/aligned.txt")

    # Plot the comparison between front and back pits
    plot(back_eval, front_eval, "Front_exposed", "Back_exposed", save=True)


# If the script is run as a standalone program, execute the comparison
if __name__ == "__main__":
    compare_CNN_aligned()

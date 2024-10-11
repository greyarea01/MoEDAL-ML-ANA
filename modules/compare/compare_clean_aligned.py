import numpy as np
from modules.dataset_tools import coordinates, mhio
import matplotlib.pyplot as plt
from modules.compare.tools import extract_coord

'''
This script compares two lists of etch pits (e.g., from different models or methods) 
and plots the comparison between them.
'''


def plot(list1, list2, name1, name2, save=True):
	"""
	Plots the coordinates of two sets of etch pits on a 2D scatter plot.

	Parameters:
	- list1, list2: Lists containing pit coordinates.
	- name1, name2: Labels for the two sets of pits to be displayed in the plot legend.
	- save: Whether to save the plot to a file (default is True).

	Returns:
	- None
	"""
	plt.figure(figsize=(15, 15), dpi=100)  # Set up the figure size and resolution
	C1 = plt.scatter(x=extract_coord(list1, "x"), y=extract_coord(list1, "y"), marker="o", s=30,
	                 color="green")  # Plot list1 pits
	Cr = plt.scatter(x=extract_coord(list2, "x"), y=extract_coord(list2, "y"), marker="x", s=30,
	                 color="blue")  # Plot list2 pits
	plt.title(name1 + " and " + name2)  # Set the plot title
	plt.xlabel("X")  # X-axis label
	plt.ylabel("Y")  # Y-axis label
	plt.legend((C1, Cr), (name1, name2), scatterpoints=1, loc=0, ncol=2, fontsize=8)  # Add a legend for both pit sets
	if save:
		plt.savefig(f"plots/clean_{name1}_and_{name2}.png", dpi=100)  # Save the plot if 'save' is True
	plt.show()  # Display the plot


def compare_clean_aligned():
	"""
	Compares two sets of etch pits (Kanik and Lewis) by calculating overlapping and non-overlapping pits,
	and saves the results to JSON files. The comparison results are also plotted.

	Returns:
	- None
	"""

	# Load pit data for Kanik and Lewis (aligned)
	kanik_global = mhio.load_json("kanik/clean_aligned.txt")  # Load Kanik's pits from JSON
	global_location = mhio.load_json(
		"modules/dataset_tools/metadata/all_aligned_Global_locations.txt")  # Load Lewis's pits (global data)

	print(global_location['help'])  # Print the help section for additional information

	# Extract Lewis's pits data
	lewis_global = global_location['foil_1_2_aligned']
	lewis_global_eval = []

	# Initialize a coordinate system to translate global locations into ixy system
	CS = coordinates.Coord_system()

	# Filter out invalid pits in Lewis's data (negative coordinates or below the threshold)
	for i in range(len(lewis_global)):
		tmp = CS['c1'].get_ixy(lewis_global[i])
		if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
			continue  # Skip invalid pits
		lewis_global_eval.append(lewis_global[i][:])  # Append valid pits to the evaluation list

	# Print the number of pits in Lewis's and Kanik's sets
	print("# of Lewis's pits: {}".format(len(lewis_global_eval)))
	print("# of Kanik's pits: {}".format(len(kanik_global)))

	# Use coordinates.Overlap to compute the overlap and non-overlap between the two pit sets
	x = coordinates.Overlap(np.array(lewis_global_eval), np.array(kanik_global), tolerance=12)

	# Tally the overlap results
	coordinates.tally(x)

	# Get the indices for only Kanik's pits, only Lewis's pits, and common pits
	only_kanik_ind = x.B_not_A()
	only_lewis_ind = x.A_not_B()
	common_ind = x.A_in_B()

	# Extract the actual pit coordinates based on the indices
	only_kanik = [kanik_global[int(only_kanik_ind[0][i])] for i in range(len(only_kanik_ind[0]))]  # Kanik's only pits
	only_lewis = [lewis_global_eval[int(only_lewis_ind[0][i])] for i in
	              range(len(only_lewis_ind[0]))]  # Lewis's only pits
	common = [lewis_global_eval[int(common_ind[0][i])] for i in
	          range(len(common_ind[0]))]  # Common pits between Kanik and Lewis

	# Print the results of the comparison
	print("Only Kanik = {}".format(len(only_kanik)))
	print("Only Lewis = {}".format(len(only_lewis)))
	print("Common = {}".format(len(common)))

	# Save the comparison results to JSON files
	mhio.save_json(only_lewis, "predicted_list/universal_true_etch_pit_list/Aligned_L_Not_K.txt")
	mhio.save_json(only_kanik, "predicted_list/universal_true_etch_pit_list/Aligned_K_Not_L.txt")
	mhio.save_json(common, "predicted_list/universal_true_etch_pit_list/Aligned_common.txt")

	# Plot the comparison results
	plot(kanik_global, lewis_global_eval, "DCC_aligned", "Laplace_aligned")


# If the script is run as the main program, execute the comparison function
if __name__ == "__main__":
	compare_clean_aligned()

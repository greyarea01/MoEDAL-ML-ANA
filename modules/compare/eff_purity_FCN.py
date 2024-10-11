"""
Get all predicted list from FCN front
Get universal true list
Find distance from the closest etch pit in the predicted list from true list
create histogram with distance
plot efficiency (pred_true_pit/total_true_pit) vs cut value
plot purity (pred_true_pit/pred_true_pit+false_pos vs cut value
"""

"""
A: etch pit in clean + etch pit in exposed within distance d
B: etch pit in clean + no etch pit in exposed within distance d 
C: etch pit in exposed + no etch pit in clean within distance d 

A = true positives
B = false negatives
C = false positives
eff = A / (A+B)
purity = A / (A+C)
"""

import numpy as np
from modules.dataset_tools import coordinates, mhio
import matplotlib.pyplot as plt
from math import ceil


def get_abc(distance_true, distance_pred):
	"""
	:param distance_true: dist of true pits from the nearest pred etch pit
	:param distance_pred: dist of pred pits from the nearest true etch pit
	:return: list a, b, c as described in the header of the same length max(distance)
	"""
	a = []
	b = []
	c = []
	for d in range(ceil(max(distance_pred))):
		a.append(sum(count < d for count in distance_true))
		b.append(sum(count > d for count in distance_true))
		c.append(sum(count > d for count in distance_pred))

	return a, b, c


def get_eff_pur(a, b, c):
	"""
	:param a: etch pit in clean + etch pit in exposed within distance d
	:param b: etch pit in clean + no etch pit in exposed within distance d
	:param c: etch pit in exposed + no etch pit in clean  within distance d
	:return: efficiency and purity curves wrt the cut value
	"""

	purity = []
	efficiency = []
	for dis in range(len(a)):
		efficiency.append(a[dis] / (a[dis] + b[dis]))
		purity.append(a[dis] / (a[dis] + c[dis]))

	return efficiency, purity

def evaluate_efficiency_purity_FCN():
	"""
	Main function to evaluate efficiency and purity for FCN front pits using the true and predicted lists.
	Returns:
	- None
	"""
	true_eval = mhio.load_json("Aligned_clean.txt")  # in Lewis's and in Kanik's
	CS = coordinates.Coord_system()

	predicted_list = mhio.load_json("predicted_etch_pits/FCN/final_pr.txt")
	predicted_list_eval = []

	for i in range(len(predicted_list)):
		tmp = CS['c1'].get_ixy(predicted_list[i])
		if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
			continue
		tmp = CS.convert(tmp, 'cd')
		if tmp[1] < 0 or tmp[2] < 0 or tmp[0] < 251:
			continue
		predicted_list_eval.append(predicted_list[i][:])

	x = coordinates.Overlap(np.array(predicted_list_eval), np.array(true_eval), tolerance=15)

	nearest_point_ind_of_all_pred = x.nearest_a
	nearest_point_in_true_of_all_pred = []
	nearest_x_y_dist_of_all_pred = []
	for i in range(len(nearest_point_ind_of_all_pred)):
		nearest_point_in_true_of_all_pred.append(true_eval[int(nearest_point_ind_of_all_pred[i])])
		nearest_x_y_dist_of_all_pred.append([abs(nearest_point_in_true_of_all_pred[i][0] - predicted_list_eval[i][0]),
		                                     abs(nearest_point_in_true_of_all_pred[i][1] - predicted_list_eval[i][1])])

	nearest_distance_of_all_pred = x.closest_dist_a
	nearest_distance_of_all_true = x.closest_dist_b

	"""
	nearest_distance_of_all_pred: nearest distance to true pit from all pred same length as predicted_list_eval
	nearest_distance_of_all_true: nearest distance to true pit from all true same length as true_eval
	predicted_list_eval: list of all the predicted pits 
	true_eval : contains our true positives 
	"""

	a, b, c = get_abc(nearest_distance_of_all_true, nearest_distance_of_all_pred)
	eff, pur = get_eff_pur(a, b, c)
	cut = np.arange(0, len(eff))
	"""
	plt.scatter(cut, eff)
	
	plt.show()
	"""

	fig, ax = plt.subplots()

	ln1 = ax.plot(cut, eff, color="orange", marker="o", markersize=3, label="Efficiency")
	ax.set_xlabel("dS", fontsize=14)
	ax.set_ylabel("Efficiency", fontsize=14)
	ax.set_xscale('log')
	ax2 = ax.twinx()

	ln2 = ax2.plot(cut, pur, color="blue", marker="o", markersize=3, label="Purity")
	ax2.set_ylabel("Purity", fontsize=14)
	plt.title("Evaluation Metrics\nFCN")
	lns = ln1 + ln2
	labs = [l.get_label() for l in lns]
	ax.legend(lns, labs, loc=0)
	plt.show()
	fig.savefig('plots/eff_purr_FCN', dpi=200, bbox_inches='tight')

# Call the main function to evaluate efficiency and purity
if __name__ == "__main__":
	evaluate_efficiency_purity_FCN()



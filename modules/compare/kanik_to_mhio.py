import numpy as np
from modules.dataset_tools import coordinates, mhio
import matplotlib.pyplot as plt
from tools import extract_coord


def plot(list1, name1, save=False):
	plt.figure(figsize=(15, 15), dpi=100)
	C1 = plt.scatter(x=extract_coord(list1, "x"), y=extract_coord(list1, "y"), marker="o", s=30, color="green")
	plt.title(name1)
	plt.xlabel("X")
	plt.ylabel("Y")
	if save:
		plt.savefig("plots/"+name1, dpi=100)
	plt.show()

'''
import kanik's txt file
get coordinates
add to predicted list
convert to global c1
save in mhio
'''
threshold = 4
x_threshold = 640 - threshold
y_theshold = 480 - threshold

predicted_list = []

fileIN = open("kanik/og_data/DirtyTestReverseCentroid.txt")
lines = fileIN.readlines()

print(len(lines))

for line in lines:
    x = line.strip().split("\t")
    try:
        x = list(map(float, x))
        if x[0] == '0' or x[1] < threshold or x[2] < threshold or x[1] > x_threshold or x[2] > y_theshold:
            continue
        predicted_list.append([float(x[0]), float(x[1]), float(x[2])])
    except Exception:
        pass


print(len(predicted_list))
cs = coordinates.Coord_system()
cs['dr'].flipped = True    # converting coordinates to global
cs['dr'].reverse = True
for i in range(len(predicted_list)):
    predicted_list[i] = cs.convert(predicted_list[i], 'drc')
    predicted_list[i] = cs['c1'].get_gxy(predicted_list[i])
    predicted_list[i] = predicted_list[i].tolist()

#plot(predicted_list, "pred_reverse_DCC")

mhio.save_json(predicted_list, "kanik/final_reverse_pr.txt")

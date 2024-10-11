from modules.dataset_tools import mhio, coordinates

CS = coordinates.Coord_system()

def clean_common_aligned(threshold=4):
    """
    Filters the common etch pit list based on a threshold, removing pits
    that are outside of the defined x and y coordinate limits.

    Parameters:
    - threshold: The margin threshold for filtering out pits close to the edges (default is 4).

    Returns:
    - None
    """

    # Set thresholds for valid coordinates
    x_threshold = 640 - threshold
    y_threshold = 480 - threshold

    # Load the pit lists
    list_1 = mhio.load_json("predicted_list/universal_true_etch_pit_list/Aligned_K_Not_L.txt")  # Kanik's only
    list_2 = mhio.load_json("predicted_list/universal_true_etch_pit_list/Aligned_L_Not_K.txt")  # Lewis's only
    list_3 = mhio.load_json("predicted_list/universal_true_etch_pit_list/Aligned_common.txt")  # Common pits

    common = list_3
    mask = []

    # Loop through the common pits and apply filtering based on coordinate thresholds
    for i in range(len(common)):
        tmp = CS['c1'].get_ixy(common[i])  # Convert to ixy coordinates
        tmp = CS.convert(tmp, 'cd')  # Convert to cd coordinates
        # Filter out pits that are too close to the edges or below the minimum depth threshold
        if tmp[1] < threshold or tmp[2] < threshold or tmp[1] > x_threshold or tmp[2] > y_threshold or tmp[0] < 251:
            continue
        mask.append(common[i][:])  # Append valid pits to the mask

    common = mask  # Update the common pit list with filtered results
    print(f"Number of valid common pits: {len(common)}")

    # Save the filtered common pit list to a JSON file
    mhio.save_json(common, "Aligned_clean.txt")

if __name__ == "__main__":
    clean_common_aligned()

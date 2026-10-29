import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from modules.dataset_tools import coordinates

# -------------------------------------------------------
# Configure global plotting style for publication-quality
# -------------------------------------------------------
plt.rcParams.update({
    'font.size':       14,      # base text size
    'font.family':     'serif', # use serif fonts for formality
    'axes.titlesize':  16,      # subplot title size
    'axes.labelsize':  14,      # axis label size
    'xtick.labelsize': 12,      # x-tick label size
    'ytick.labelsize': 12,      # y-tick label size
    'legend.fontsize': 12,      # legend text size
    'figure.dpi':      150      # higher DPI for crisp output
})

def crop_image_with_boundary_and_padding(img, x, y, crop_size):
    """
    Extract a square patch centered at (x,y) from a 2D image, padding with zeros
    if the patch extends beyond the image borders.

    Parameters:
        img (np.ndarray): 2D array (height × width).
        x (int): x-coordinate of the center pixel.
        y (int): y-coordinate of the center pixel.
        crop_size (int): length of each side of the square patch.

    Returns:
        np.ndarray: cropped patch of shape (crop_size, crop_size).
    """
    half = crop_size // 2

    # clamp edges so we don't index out of bounds
    x0 = max(0, x - half)
    x1 = min(img.shape[1], x + half)
    y0 = max(0, y - half)
    y1 = min(img.shape[0], y + half)
    patch = img[y0:y1, x0:x1]

    # compute zero-padding amounts on each side
    pad_left   = max(0, half - x)
    pad_right  = max(0, (x + half) - img.shape[1])
    pad_top    = max(0, half - y)
    pad_bottom = max(0, (y + half) - img.shape[0])

    # pad with zeros to restore full square shape
    return np.pad(
        patch,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )


def create_example_panel(
        global_coord=None,
        save_folder="gallery",
        save_name=None,
        title="Image Panel",
        description=None,
        extra_fcn_coords=None,
        extra_dcc_coords=None,
        extra_fcn_coords_front=None,
        extra_dcc_coords_front=None,
        extra_fcn_coords_back=None,
        extra_dcc_coords_back=None,
        clean_front_path="numpy_data_eval/preprocessed_clean_eval.npy",
        clean_back_path="numpy_data_eval/preprocessed_clean_reverse_eval.npy",
        exposed_front_path="numpy_data_eval/preprocessed_dirty_eval.npy",
        exposed_back_path="numpy_data_eval/preprocessed_dirty_reverse_eval.npy",
        crop_size=56
    ):
    """
    Generate and save a 2×2 grid of cropped image patches, each framed individually,
    with overlaid markers, a shared legend, and an optional description.

    Parameters:
        global_coord (tuple): (x, y) coordinate for the main crop center.
                              If None, the first extra FCN or DCC coord is used, but nothing is plotted
        save_folder (str): directory to write the PNG into (created if needed).
        save_name (str):   filename (without “.png”).
                          If None, defaults to "example_plot_{x}_{y}".
        title (str):       super-title text (or None to omit).
        description (str): text below the legend (or None to omit).
        extra_fcn_coords (list of tuples): optional blue–× markers.
        extra_dcc_coords (list of tuples): optional green–× markers.
        *_path (str):      paths to the four .npy volumes.
        crop_size (int):   side length in pixels of each square crop.

    Returns:
        None  (writes “{save_folder}/{save_name}.png”)
    """
    os.makedirs(save_folder, exist_ok=True)
    primary_plot = True

    # 1) choose a fallback main coordinate if not explicitly provided
    if global_coord is None:
        primary_plot=False
        if extra_fcn_coords:
            global_coord = extra_fcn_coords[0]
        elif extra_dcc_coords:
            global_coord = extra_dcc_coords[0]
        else:
            raise ValueError("Need at least one coordinate to center the crops.")

    # 2) load the four 3D volumes (select slice 8 for the 4D arrays)
    image_data = {
        'clean_front':   np.load(clean_front_path),
        'clean_back':    np.load(clean_back_path),
        'exposed_front': np.load(exposed_front_path)[:, :, :, 8],
        'exposed_back':  np.load(exposed_back_path)[:, :, :, 8],
    }

    # 3) set up coordinate converter for indexing each volume
    CS = coordinates.Coord_system()
    CS['c1'].flipped = False; CS['c1'].reverse = False
    CS['cr'].flipped = True;  CS['cr'].reverse = True
    CS['dr'].flipped = True;  CS['dr'].reverse = True

    # 4) map the global (x,y) into (slice, x, y) for each volume
    cf_ixy = CS['c1'].get_ixy(global_coord)
    cb_ixy = CS.convert(cf_ixy, 'ccr0')
    ef_ixy = CS.convert(cf_ixy, 'cd')
    eb_ixy = CS.convert(cf_ixy, 'cdr')

    # 5) correct each slice‐index by subtracting the dataset‐specific offset
    cf_ixy = (cf_ixy[0] - 251, cf_ixy[1], cf_ixy[2])
    cb_ixy = (cb_ixy[0] - 251, cb_ixy[1], cb_ixy[2])
    ef_ixy = (ef_ixy[0] - 251, ef_ixy[1], ef_ixy[2])
    eb_ixy = (eb_ixy[0] - 251, eb_ixy[1], eb_ixy[2])

    local_coords = {
        'clean_front':   cf_ixy,
        'clean_back':    cb_ixy,
        'exposed_front': ef_ixy,
        'exposed_back':  eb_ixy,
    }

    # 6) crop each 2D slice around its center point
    cropped = {}
    for key, vol in image_data.items():
        i, x, y = local_coords[key]
        cropped[key] = crop_image_with_boundary_and_padding(vol[i], x, y, crop_size)

    # 7) decide grid layout: two image rows, plus a third row only if description is set
    add_desc = (description is not None)
    nrows = 2 + int(add_desc)
    # bottom row height: just enough for legend (and a bit extra if a description follows)
    height_ratios = [1, 1] + ([0.06] if add_desc else [])

    fig = plt.figure(figsize=(8, 8))
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2,
        height_ratios=height_ratios,
        hspace=0.1,
        wspace=0.1
    )

    # 8) place the four image axes in the top two rows
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    axes = [ax00, ax01, ax10, ax11]
    names = ['exposed_front', 'exposed_back', 'clean_front', 'clean_back']

    # 9) render each cropped patch, hide axes
    for ax, nm in zip(axes, names):
        ax.imshow(cropped[nm], cmap='gray')
        ax.axis('off')

    # 10) annotate top row with titles, bottom row with labels
    ax00.set_title('Exposed front')
    ax01.set_title('Exposed back')
    ax10.set_title('Clean front')
    ax11.set_title('Clean back')

    # 11) marker style definitions
    primary_kw = {
        'marker': 'o',
        'color': '#D55E00',  # vermillion/red (OKABE-ITO palette)
        'linestyle': 'None',
        'markersize': 8,
        'label': 'Primary'
    }

    fcn_kw = {
        'marker': 's',
        'markerfacecolor': 'none',
        'color': '#56B4E9',  # sky blue
        'linestyle': 'None',
        'markersize': 10,
        'markeredgewidth': 1.4,
        'label': 'FCN'
    }

    dcc_kw = {
        'marker': r'$\triangle$',
        'color': '#009E73',  # bluish green
        'linestyle': 'None',
        'markersize': 12,
        'markeredgewidth': 1,
        'label': 'DCC'
    }
    center     = crop_size // 2

    # 12) overlay the primary and any extra coordinates on each subplot
    i = 0
    for ax in axes:
        i += 1
        if primary_plot:
            ax.plot(center, center, **primary_kw)
        if extra_fcn_coords:
            for c in extra_fcn_coords:
                ex_ixy = CS['c1'].get_ixy(c)
                ex = ex_ixy[1] - cf_ixy[1] + center
                ey = ex_ixy[2] - cf_ixy[2] + center
                ax.plot(ex, ey, **fcn_kw)
        elif extra_fcn_coords_front and extra_fcn_coords_back:
            if i%2==1:
                for c in extra_fcn_coords_front:
                    ex_ixy = CS['c1'].get_ixy(c)
                    ex = ex_ixy[1] - cf_ixy[1] + center
                    ey = ex_ixy[2] - cf_ixy[2] + center
                    ax.plot(ex, ey, **fcn_kw)
            if i%2==0:
                for c in extra_fcn_coords_back:
                    ex_ixy = CS['c1'].get_ixy(c)
                    ex = ex_ixy[1] - cf_ixy[1] + center
                    ey = ex_ixy[2] - cf_ixy[2] + center
                    ax.plot(ex,ey,**fcn_kw)
        if extra_dcc_coords:
            for c in extra_dcc_coords:
                ex_ixy = CS['c1'].get_ixy(c)
                ex = ex_ixy[1] - cf_ixy[1] + center
                ey = ex_ixy[2] - cf_ixy[2] + center
                ax.plot(ex, ey, **dcc_kw)
        elif  extra_dcc_coords_front and extra_dcc_coords_back:
            if i%2==1:
                for c in extra_dcc_coords_front:
                    ex_ixy = CS['c1'].get_ixy(c)
                    ex = ex_ixy[1] - cf_ixy[1] + center
                    ey = ex_ixy[2] - cf_ixy[2] + center
                    ax.plot(ex, ey, **dcc_kw)
            if i%2==0:
                for c in extra_dcc_coords_back:
                    ex_ixy = CS['c1'].get_ixy(c)
                    ex = ex_ixy[1] - cf_ixy[1] + center
                    ey = ex_ixy[2] - cf_ixy[2] + center
                    ax.plot(ex,ey,**dcc_kw)
    # 13) add a thin black rectangle around each subplot
    for ax in axes:
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                         fill=False, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)

    # 14) shared legend across the bottom
    handles = [Line2D([], [], **primary_kw),
               Line2D([], [], **fcn_kw),
               Line2D([], [], **dcc_kw)]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.02),
               bbox_transform=fig.transFigure)

    # 15) optional description text under the legend
    if description:
        fig.text(0.5, 0.005, description,
                 ha='center', va='bottom', wrap=True, fontsize=12)

    # 16) compose filename, save, and clean up
    if save_name:
        fname = save_name
    else:
        fname = f"example_plot_{global_coord[0]}_{global_coord[1]}"
    out_path = os.path.join(save_folder, f"{fname}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Panel image saved to {out_path}")


create_example_panel(
    global_coord=(120, 240),
    save_folder="my_panels",
    save_name="panel_120_240",
    title="Sample Imaging Panel",
    description="Top row: exposed scans; bottom row: clean scans.",
    extra_fcn_coords=[(130,250), (110,230)],
    extra_dcc_coords=[(125,245)]
)

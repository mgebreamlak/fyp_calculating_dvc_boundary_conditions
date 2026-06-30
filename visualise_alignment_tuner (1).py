"""
visualise_alignment_tuner.py

Interactive DVC alignment tuner. You drag sliders to rotate the DVC point
cloud relative to the Marc mesh, watch the alignment update live in a single
3D panel, and read off the rotation values to paste back into
create_surface_nodes_xyz.py.

How to use:
  1) Run the script.
  2) Use the Rx, Ry, Rz sliders (in degrees) to rotate the DVC cloud.
     The rotation is applied about MARC_LANDMARK, exactly like the main
     pipeline.
  3) Drag the 3D panel with left mouse to look around.
  4) The current Rx, Ry, Rz values are shown live at the bottom of the
     window -- copy these into create_surface_nodes_xyz.py.
  5) Close the window when done -- the final values are printed and a
     snapshot PNG is saved.

The DVC point cloud is COLOUR-CODED by displacement magnitude (column 4 of the
tecplot file -- the 'w' value). A colorbar is added on the right.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation

dat_file       = r"C:\Users\rahwa.tecle\OneDrive - Imperial College London\FYP\Mesh Only\250matprop.dat"
dvc_nodes_file = r"C:\Users\rahwa.tecle\OneDrive - Imperial College London\FYP\B0001.dat"
output_png     = r"C:\Users\rahwa.tecle\OneDrive - Imperial College London\FYP\Mesh Only\alignment_tuner_snapshot.png"

top_set_name    = "top_surface_nodes"
bottom_set_name = "bottom_surface_nodes"

PIXEL_SIZE = 0.039

DVC_LANDMARK_PX = np.array([342.0, 304.0, 532.0])
MARC_LANDMARK   = np.array([13.41223140, -95.48385620, -1220.89936])

# Starting values for the sliders
Rx0, Ry0, Rz0 = 0.0, 22.0, -84.5

# Starting view angle for the 3D panel
VIEW_ELEV = 30
VIEW_AZIM = -60

# Slider range
ROT_RANGE_DEG = 190.0

# ---- Plot styling ----
# Marc mesh -- darker greys, slightly higher alpha for better contrast
MARC_TOP_COLOR    = "#546E7A"
MARC_BOTTOM_COLOR = "#78909C"
MARC_ALPHA        = 0.35
MARC_SIZE         = 1.8

DVC_ALPHA  = 0.85
DVC_SIZE   = 6

# Colormap for DVC displacement
DVC_CMAP = "viridis"  
DVC_CLIP_PERCENTILES = (2.0, 98.0)

# Slider styling
SLIDER_TRACK_COLOR  = "#E0E0E0"
SLIDER_FILL_COLOR   = "#1976D2"
SLIDER_LABEL_COLOR  = "#37474F"

# Axis / pane styling
PANE_COLOR     = "#FAFAFA"  
GRID_COLOR     = "#E0E0E0"  
AXIS_LINE_COLOR = "#B0BEC5" 
TEXT_COLOR     = "#37474F"  
FIG_BG         = "#FFFFFF"  

FLATTEN_BY_PCA = True
# ============================


# Marc parsing
FLOAT_TOKEN_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[+-]\d+)")
ID_LINE_RE = re.compile(r"^\s*(\d+)\s*(.*)$")


def parse_marc_float(tok):
    m = re.fullmatch(r"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+))([+-]\d+)", tok.strip())
    if not m:
        raise ValueError(f"Not a Marc float: {tok!r}")
    return float(m.group(1) + "e" + m.group(2))


def extract_coordinates(filename):
    coords = {}
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            if line.strip().lower() == "coordinates":
                header = next(f)
                parts = header.split()
                npoints = int(parts[1])
                for _ in range(npoints):
                    l = next(f).rstrip("\n")
                    m = ID_LINE_RE.match(l)
                    nid = int(m.group(1))
                    rest = m.group(2)
                    toks = FLOAT_TOKEN_RE.findall(rest)
                    if len(toks) != 3:
                        raise ValueError(f"Bad line: {l!r}")
                    x, y, z = (parse_marc_float(t) for t in toks)
                    coords[nid] = (x, y, z)
                break
    return coords


def extract_nodes_from_set(filename, set_name):
    ids = []
    start_re = re.compile(rf"^\s*define\s+\w+\s+set\s+{re.escape(set_name)}\s*$", re.IGNORECASE)
    with open(filename, "r", errors="ignore") as f:
        in_set = False
        for line in f:
            if not in_set:
                if start_re.match(line):
                    in_set = True
                continue
            if line.strip() == "":
                break
            low = line.lstrip().lower()
            if low.startswith("define") or low.startswith("end"):
                break
            found = re.findall(r"\d+", line)
            if not found:
                break
            ids.extend(int(x) for x in found)
    return ids


def extract_tecplot_points(filename):
    points = []
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith(("TITLE", "VARIABLES", "ZONE", "STRANDID", "SOLUTIONTIME")):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
                w = float(parts[3]) if len(parts) >= 4 else None
                is_valid = int(float(parts[4])) if len(parts) >= 5 else None
            except ValueError:
                continue
            points.append((x, y, z, w, is_valid))
    return points


def subsample(arr, n=4000, seed=0):
    if len(arr) <= n:
        return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(len(arr), n, replace=False)]


def subsample_with_values(arr, values, n=5000, seed=0):
    """Subsample arr and values together with a shared index."""
    if len(arr) <= n:
        return arr, values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), n, replace=False)
    return arr[idx], values[idx]


def compute_flatten_transform(marc_pts):
    centroid = marc_pts.mean(axis=0)
    centered = marc_pts - centroid
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt.copy()
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return R, centroid, s


def apply_flatten(pts, R, centroid):
    return (pts - centroid) @ R.T


def style_3d_axes(ax):
    """ Replaces the default grey panes and dark gridlines with near-white panes,
    light grey gridlines, and subtle axis spines.
    """
    # Panes -- near-white, no border
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_edgecolor(AXIS_LINE_COLOR)
        axis.pane.set_alpha(1.0)

    # Gridlines -- light grey, thin, slightly transparent
    ax.xaxis._axinfo["grid"].update(color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.yaxis._axinfo["grid"].update(color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.zaxis._axinfo["grid"].update(color=GRID_COLOR, linewidth=0.6, alpha=0.7)

    # Tick labels -- dark slate, slightly smaller, no bold default look
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=8, pad=2)
    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=8, pad=2)
    ax.tick_params(axis="z", colors=TEXT_COLOR, labelsize=8, pad=2)

    # Axis label styling
    for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        label.set_color(TEXT_COLOR)
        label.set_fontsize(10)
        label.set_fontweight("medium")

    # Title styling
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("medium")


def style_slider(slider):
    slider.track.set_facecolor(SLIDER_TRACK_COLOR)
    slider.track.set_edgecolor("none")
    slider.poly.set_facecolor(SLIDER_FILL_COLOR)
    slider.poly.set_edgecolor("none")
    slider.poly.set_alpha(0.85)
    # The little vertical "handle" line at the current value
    try:
        slider.hline.set_color(SLIDER_FILL_COLOR)
        slider.hline.set_linewidth(0)  # hide -- the fill itself indicates value
    except AttributeError:
        pass
    # Label and value text
    slider.label.set_fontsize(10)
    slider.label.set_color(SLIDER_LABEL_COLOR)
    slider.label.set_fontweight("medium")
    slider.valtext.set_fontsize(9)
    slider.valtext.set_color(SLIDER_LABEL_COLOR)
    slider.valtext.set_fontfamily("monospace")
    # Hide the slider's own axes box -- the track shape provides all the visual
    ax = slider.ax
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")


def main():
    # ---- Load Marc ----
    print("Loading Marc mesh...")
    coords = extract_coordinates(dat_file)
    top_ids = extract_nodes_from_set(dat_file, top_set_name)
    bot_ids = extract_nodes_from_set(dat_file, bottom_set_name)
    top = np.array([coords[n] for n in top_ids if n in coords])
    bot = np.array([coords[n] for n in bot_ids if n in coords])
    all_marc = np.vstack([top, bot])

    # ---- Load DVC raw (pre-rotation) ----
    print("Loading DVC...")
    raw = extract_tecplot_points(dvc_nodes_file)
    raw_arr = np.array([(p[0], p[1], p[2],
                         p[3] if p[3] is not None else 0.0,
                         p[4] if p[4] is not None else 0) for p in raw])
    swapped = np.column_stack([
        raw_arr[:, 0] * PIXEL_SIZE,
        raw_arr[:, 1] * PIXEL_SIZE,
        raw_arr[:, 2] * PIXEL_SIZE,
    ])
    valid = raw_arr[:, 4] == 1

    dvc_lm_postswap_mm = np.array([
        DVC_LANDMARK_PX[0] * PIXEL_SIZE,
        DVC_LANDMARK_PX[1] * PIXEL_SIZE,
        DVC_LANDMARK_PX[2] * PIXEL_SIZE,
    ])
    translation_lm = MARC_LANDMARK - dvc_lm_postswap_mm

    dvc_pre_rot = swapped[valid] + translation_lm
    # Column 4 ('w') is displacement in PIXELS -- convert to mm to match the
    # rest of the pipeline.
    dvc_disp = raw_arr[valid, 3].astype(float) * PIXEL_SIZE
    print(f"  valid points: {len(dvc_pre_rot)}")
    print(f"  displacement range: [{dvc_disp.min():.4g}, {dvc_disp.max():.4g}] mm")
    print(f"  displacement mean +/- std: {dvc_disp.mean():.4g} +/- {dvc_disp.std():.4g} mm")

    dvc_pre_rot, dvc_disp = subsample_with_values(dvc_pre_rot, dvc_disp, n=5000)

    # ---- Colour normalisation for DVC displacement ----
    lo_pct, hi_pct = DVC_CLIP_PERCENTILES
    vmin = float(np.percentile(dvc_disp, lo_pct))
    vmax = float(np.percentile(dvc_disp, hi_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    print(f"  colour scale clipped to [{vmin:.4g}, {vmax:.4g}] mm "
          f"(percentiles {lo_pct}-{hi_pct})")
    dvc_norm = Normalize(vmin=vmin, vmax=vmax)
    dvc_cmap = plt.get_cmap(DVC_CMAP)
    dvc_facecolors = dvc_cmap(dvc_norm(dvc_disp))

    # ---- PCA flatten ----
    if FLATTEN_BY_PCA:
        R_flat, centroid_flat, sing = compute_flatten_transform(all_marc)
        print(f"PCA flatten sigmas: {sing[0]:.2f}, {sing[1]:.2f}, {sing[2]:.2f} mm")
        top_v = apply_flatten(top, R_flat, centroid_flat)
        bot_v = apply_flatten(bot, R_flat, centroid_flat)
        axis_label_suffix = " (PCA frame)"
    else:
        top_v = top
        bot_v = bot
        axis_label_suffix = ""

    top_s = subsample(top_v, n=5000)
    bot_s = subsample(bot_v, n=5000)

    def transform_dvc(Rx, Ry, Rz):
        rot = Rotation.from_euler("xyz", [Rx, Ry, Rz], degrees=True)
        out = rot.apply(dvc_pre_rot - MARC_LANDMARK) + MARC_LANDMARK
        if FLATTEN_BY_PCA:
            out = apply_flatten(out, R_flat, centroid_flat)
        return out

    dvc_now = transform_dvc(Rx0, Ry0, Rz0)

    # ---- Figure layout (single 3D viewport) ----
    fig = plt.figure(figsize=(12, 10), facecolor=FIG_BG)
    fig.suptitle(
        "DVC Alignment Tuner",
        fontsize=13, fontweight="semibold", color=TEXT_COLOR, y=0.96
    )
    fig.text(
        0.5, 0.925,
        "Adjust Rx / Ry / Rz to align DVC to Marc  ·  DVC coloured by displacement",
        ha="center", va="top",
        fontsize=9, color=TEXT_COLOR, alpha=0.75,
    )
    plt.subplots_adjust(left=0.05, right=0.86, top=0.90, bottom=0.24)

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Marc (static) -- two scatters for top/bottom colours, but only one
    # legend entry since they're the same mesh logically
    ax.scatter(top_s[:, 0], top_s[:, 1], top_s[:, 2],
               s=MARC_SIZE, c=MARC_TOP_COLOR, alpha=MARC_ALPHA,
               label="Marc mesh", depthshade=False)
    ax.scatter(bot_s[:, 0], bot_s[:, 1], bot_s[:, 2],
               s=MARC_SIZE, c=MARC_BOTTOM_COLOR, alpha=MARC_ALPHA,
               label="_nolegend_", depthshade=False)

    # DVC scatter (updated by sliders)
    sc = ax.scatter(dvc_now[:, 0], dvc_now[:, 1], dvc_now[:, 2],
                    s=DVC_SIZE, c=dvc_facecolors, alpha=DVC_ALPHA,
                    label="DVC valid", depthshade=False)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    try:
        ranges = np.ptp(np.vstack([top_s, bot_s, dvc_now]), axis=0)
        ax.set_box_aspect(ranges)
    except Exception:
        pass

    # Apply clean axis / pane / grid styling
    style_3d_axes(ax)

    # Legend -- frameless, light, top-left
    leg = ax.legend(loc="upper left", markerscale=1.5, fontsize=9,
                    frameon=False, labelcolor=TEXT_COLOR)

    # Colorbar
    sm = cm.ScalarMappable(norm=dvc_norm, cmap=dvc_cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.89, 0.40, 0.02, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DVC displacement (mm)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    # Sliders (rotation only)
    slider_specs = [
        ("Rx", Rx0, -ROT_RANGE_DEG, ROT_RANGE_DEG),
        ("Ry", Ry0, -ROT_RANGE_DEG, ROT_RANGE_DEG),
        ("Rz", Rz0, -ROT_RANGE_DEG, ROT_RANGE_DEG),
    ]
    sliders = []

    slider_left = 0.20
    slider_w    = 0.50
    slider_h    = 0.018 
    slider_gap  = 0.025
    base_y      = 0.135  

    for i, (label, init, lo, hi) in enumerate(slider_specs):
        y = base_y - i * slider_gap
        ax_s = fig.add_axes([slider_left, y, slider_w, slider_h])
        s = Slider(ax_s, label, lo, hi, valinit=init, valstep=0.5,
                   valfmt="%6.1f deg")
        style_slider(s)
        sliders.append(s)

    s_Rx, s_Ry, s_Rz = sliders

    def on_slider_change(_val):
        new_dvc = transform_dvc(s_Rx.val, s_Ry.val, s_Rz.val)
        sc._offsets3d = (new_dvc[:, 0], new_dvc[:, 1], new_dvc[:, 2])
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(on_slider_change)

    def on_close(_event):
        print("\n" + "=" * 60)
        print("FINAL ALIGNMENT VALUES (window closed)")
        print("=" * 60)
        print(f"Rx, Ry, Rz = {s_Rx.val:.2f}, {s_Ry.val:.2f}, {s_Rz.val:.2f}")
        print("=" * 60)
        try:
            fig.savefig(output_png, dpi=140, bbox_inches="tight")
            print(f"Snapshot saved: {output_png}")
        except Exception as e:
            print(f"Could not save snapshot: {e}")

    fig.canvas.mpl_connect("close_event", on_close)

    print("\nWindow open.")
    print("  - Drag Rx / Ry / Rz sliders to rotate the DVC cloud.")
    print("  - Drag the 3D panel to change view angle.")
    print("  - Current values are shown live at the bottom of the window.")
    print("  - Close the window to print final values + save snapshot.\n")
    plt.show()


if __name__ == "__main__":
    main()

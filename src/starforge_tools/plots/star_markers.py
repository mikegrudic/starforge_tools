import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

star_colors = (
    np.array([[255, 203, 132], [255, 243, 233], [155, 176, 255]]) / 255
)  # default colors, reddish for small ones, yellow-white for mid sized and blue for large


def plot_star_markers(ax, pdata):
    """Plots star markers with sizes and colors varying according to stellar mass.
    Each marker is randomly rotated (deterministic seed for reproducibility)."""
    xs = pdata["PartType5/Coordinates"]
    ms = pdata["PartType5/BH_Mass"]
    order = ms.argsort()
    xs, ms = xs[order], ms[order]
    starcolors = np.array([np.interp(np.log10(ms), [-1, 0, 1], star_colors[:, i]) for i in range(3)]).T
    sizes = 4 * (ms / 1) ** 0.5
    angles = np.random.default_rng(0).uniform(0, 360, size=len(ms))
    for i in range(len(ms)):
        marker = MarkerStyle("*", transform=Affine2D().rotate_deg(angles[i]))
        ax.scatter(
            xs[i, 0],
            xs[i, 1],
            s=sizes[i],
            edgecolor="black",
            facecolor=starcolors[i],
            marker=marker,
            lw=0.02,
            alpha=0.5,
        )


def plot_star_legend(ax):
    """Plots the legend for stellar masses for star markers"""
    for m_dummy in 0.1, 1, 10, 100, 1000:
        ax.scatter(
            [np.inf],
            [np.inf],
            s=4 * (m_dummy) ** (0.5),
            color=[np.interp(np.log10(m_dummy), [-1, 0, 1], star_colors[:, i]) for i in range(3)],
            label=r"$%gM_\odot$" % m_dummy,
            edgecolor="black",
            alpha=1,
            lw=0.02,
            marker="*",
        )
    ledge = ax.legend(loc=2, frameon=True, facecolor="black", labelspacing=0.1, fontsize=6, edgecolor="white")
    ledge.get_frame().set_linewidth(0.5)
    ledge.get_frame().set_alpha(0.5)
    for text in ledge.get_texts():
        text.set_color("white")

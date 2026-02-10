import numpy as np

star_colors = (
    np.array([[255, 203, 132], [255, 243, 233], [155, 176, 255]]) / 255
)  # default colors, reddish for small ones, yellow-white for mid sized and blue for large


def plot_star_markers(ax, pdata):
    """Plots star markers with sizes and colors varying according to stellar mass."""
    xs = pdata["PartType5/Coordinates"]
    ms = pdata["PartType5/BH_Mass"]
    xs, ms = xs[ms.argsort()][::-1], np.sort(ms)[::-1]
    starcolors = np.array([np.interp(np.log10(ms), [-1, 0, 1], star_colors[:, i]) for i in range(3)]).T
    ax.scatter(
        xs[:, 0],
        xs[:, 1],
        s=4 * (ms / 1) ** (0.5),
        edgecolor="black",
        facecolor=starcolors,
        marker="*",
        lw=0.02,
        alpha=1,
    )


def plot_star_legend(ax):
    """Plots the legend for stellar masses for star markers"""
    for m_dummy in 0.1, 1, 10, 100:
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

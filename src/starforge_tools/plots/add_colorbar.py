import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm as norm


def my_log_formatter(x, y):
    """inspired by the nightmare mess that Jonathan Stern
    sent me after being offended by my ugly log axes"""
    if x in [1e-2, 1e-1, 1, 10, 100]:
        return r"$%g$" % x
    elif 1e-2 < x < 100 and np.isclose(0, (x * 100) % 1):
        return r"$%g$" % x
    else:
        return mpl.ticker.LogFormatterMathtext()(x)


star_colors = (
    np.array([[255, 203, 132], [255, 243, 233], [155, 176, 255]]) / 255
)  # default colors, reddish for small ones, yellow-white for mid sized and blue for large


def get_lic(vx, vy, nkern=31, trim=True, sharpen=False):
    # Generate field lines for vector field (vx,vy) using line integral-convolution, based on visualization script for FIRE from Philip Hopkins
    from licplot import (
        lic_internal,
    )  # only import line integral-convolution module if used, can be installed as pip install licplot

    texture = np.random.rand(vx.shape[0], vx.shape[1])
    x = np.arange(nkern) / nkern
    kernel = np.sin(np.pi * x) * np.pi / (2.0 * nkern)
    image_one = lic_internal.line_integral_convolution(
        vx.astype(np.float32),
        vy.astype(np.float32),
        texture.astype(np.float32),
        kernel.astype(np.float32),
    )
    image = image_one
    if trim:
        # enhance contrast
        # image_norm = (image-np.min(image))/np.ptp(image)
        # print(np.mean(image_norm), np.median(image_norm), np.std(image_norm), np.percentile(image_norm,10),np.percentile(image_norm,25),np.percentile(image_norm,50),np.percentile(image_norm,75),np.percentile(image_norm,90))
        vmin = np.mean(image_one) - np.std(image_one)
        vmax = np.mean(image_one) + np.std(image_one)
        im_trim = (image_one - vmin) / (vmax - vmin)
        im_trim[(im_trim >= 1)] = 1
        im_trim[(im_trim <= 0)] = 0
        im_trim[(np.isnan(im_trim))] = 0
        image = im_trim
    #     if sharpen:
    #         # sharpen image
    #         alpha=0.5; laplacian = (4/(alpha+1)) * np.array([ [alpha/4, (1-alpha)/4, alpha/4], [(1-alpha)/4, -1, (1-alpha)/4], [alpha/4, (1-alpha)/4, alpha/4] ])
    #         image_sharpener = convolve(im_trim, laplacian, mode='nearest')
    #         image_sharp = im_trim - image_sharpener
    #         image_sharp[(image_sharp<0)]=0; image_sharp[(image_sharp>1)]=1;
    #         image = image_sharp
    #         # re-process with a new round of LIC
    #         nkern = np.round(nkern/8.).astype('int')
    #         if(nkern<4): nkern=4;
    #         x = np.arange(nkern) / nkern;
    #         kernel = np.sin(np.pi * x) * np.pi/(2.*nkern);
    #         image = lic_internal.line_integral_convolution(vx.astype(np.float32),vy.astype(np.float32), image_sharp.astype(np.float32), kernel.astype(np.float32))
    return image


LIC_map_max_alpha = 0.5


def get_lic_image(vx, vy):
    kernel_length = 4 * int(vx.shape[0] / 1024) * 32 - 1
    image = get_lic(vx, vy, nkern=kernel_length, sharpen=False)  # get LIC image
    image_color = mpl.colors.Normalize()(image)
    image_color = plt.cm.Greys(image_color)
    image_color[..., -1] = LIC_map_max_alpha * (image - np.min(image)) / np.ptp(image)  # set transparency
    return image_color


def addColorbar(
    ax,
    cmap,
    vmin,
    vmax,
    label,
    logflag=1,
    fontsize=10,
    cmap_number=0,
    tick_tuple=None,
    horizontal=False,
    span_full_figure=True,
    nticks=5,
):
    if logflag:
        #        from mpl.colors import LogNorm as norm

        ticks = np.linspace(np.log10(vmin), int(np.log10(vmax)), nticks, endpoint=True)
        ticks = 10**ticks
        tick_labels = [my_log_formatter(tick, None) for tick in ticks]
    else:
        #   from mpl.colors import Normalize as norm

        tick_labels = [my_log_formatter(tick, None) for tick in ticks]
        ticks = np.linspace(vmin, vmax, nticks, endpoint=True)
        tick_labels = ["%.2f" % tick for tick in ticks]

    if tick_tuple is not None:
        ticks, tick_labels = tick_tuple

    fig = ax.get_figure()
    ## x,y of bottom left corner, width,height in percentage of figure size
    ## matches the default aspect ratio of matplotlib
    cur_size = fig.get_size_inches() * fig.dpi
    bbox = ax.get_position()
    extents = bbox.extents
    offset = 0  # 10
    if span_full_figure:
        for ax in fig.get_axes():
            bbox = ax.get_position()
            these_extents = bbox.extents
            for i in range(2):
                if these_extents[i] < extents[i]:
                    extents[i] = these_extents[i]
            for i in range(2, 4):
                if these_extents[i] > extents[i]:
                    extents[i] = these_extents[i]
        height = extents[3] - extents[1]
        width = extents[2] - extents[0]
    else:
        height = bbox.height
        width = bbox.width

    fig_x0, fig_y0, fig_x1, fig_y1 = extents

    if not horizontal:
        thickness = 20.0 / cur_size[0] * fig.dpi / 100
        ax1 = fig.add_axes([fig_x1 + offset / cur_size[0], fig_y0, thickness, height])

    else:
        xlabel = ax.xaxis.get_label()
        if xlabel.get_text() != "":
            print("addColorbar does not support finding xaxis text, this will look bad")
        thickness = 20.0 / cur_size[1] * fig.dpi / 100
        ax1 = fig.add_axes([fig_x0, fig_y0 - thickness - offset / cur_size[1], width, thickness])

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    cb1 = mpl.colorbar.ColorbarBase(
        ax1,
        cmap=cmap,
        # extend='both',
        # extendfrac=0.05,
        norm=norm(vmin=vmin, vmax=vmax),
        orientation="vertical" if not horizontal else "horizontal",
    )

    cb1.set_label(label, fontsize=fontsize)

    cb1.set_ticks(ticks)
    if tick_labels is not None:
        cb1.set_ticklabels(tick_labels)
    cb1.ax.tick_params(labelsize=fontsize)
    return cb1, ax1

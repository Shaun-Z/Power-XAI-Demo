import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.sparse
from shap import Explanation
from shap.utils import format_value, safe_isinstance, OpChain
from shap.utils._exceptions import DimensionError
from shap.plots import colors
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    merge_nodes,
    sort_inds,
)

labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    # 'VALUE': "SHAP value (impact on model output)",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}


def waterfall(shap_values, max_display=6, row_height = 0.2, show=True):
    """Plots an explanation of a single prediction as a waterfall plot.

    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features.

    Features are sorted by the magnitude of their SHAP values with the smallest
    magnitude features grouped together at the bottom of the plot when the number of
    features in the models exceeds the ``max_display`` parameter.

    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional :class:`.Explanation` object that contains the feature values and SHAP values to plot.

    max_display : int
        The maximum number of features to display (default is 10).
    
    row_height : float
        The height of each row in the plot (default is 0.2).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it
        has been created, returning the current axis via plt.gca().

    Examples
    --------
    See `waterfall plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html>`_.

    """
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure the input is an Explanation object
    if not isinstance(shap_values, Explanation):
        emsg = (
            "The waterfall plot requires an `Explanation` object as the "
            "`shap_values` argument."
        )
        raise TypeError(emsg)

    # Normalize shap_values.data to the range 0.95-1.05
    min_val = shap_values.data.min()
    max_val = shap_values.data.max()

    shap_values.data = 0.97 + (shap_values.data - min_val) * (1.03 - 0.97) / (max_val - min_val)

    # make sure we only have a single explanation to plot
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        emsg = (
            "The waterfall plot can currently only plot a single explanation, but a "
            f"matrix of explanations (shape {sv_shape}) was passed! Perhaps try "
            "`shap.plots.waterfall(shap_values[0])` or for multi-output models, "
            "try `shap.plots.waterfall(shap_values[0, 0])`."
        )
        raise ValueError(emsg)

    base_values = float(shap_values.base_values)
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # unwrap pandas series
    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                     color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = format_value(float(features[order[i]]), "%0.03f") + " = " + feature_names[order[i]]
            else:
                yticklabels[rng[i]] = str(features[order[i]]) + " = " + str(feature_names[order[i]])

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other buses" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
        list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
             left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
             left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb, width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=colors.light_red_rgb,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb, width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=colors.light_blue_rgb,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8)
    plt.yticks(ytick_pos, yticklabels[:-1] + [label.split('=')[-1] for label in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([base_values, base_values+1e-8])  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(["$f(x)$", "$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))

    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
    else:
        return plt.gca()

def beeswarm(shap_values, max_display=10, order=Explanation.abs.mean(0),
             clustering=None, cluster_threshold=0.5, color=None,
             axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, s=16, plot_size="auto", color_bar_label=labels["FEATURE_VALUE"]):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further
        after it has been created, returning the current axis via plt.gca().

    color_bar : bool
        Whether to draw the color bar (legend).

    s : float
        What size to make the markers. For further information see `s` in ``matplotlib.pyplot.scatter``.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged.

    Examples
    --------
    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """
    if not isinstance(shap_values, Explanation):
        emsg = (
            "The beeswarm plot requires an `Explanation` object as the "
            "`shap_values` argument."
        )
        raise TypeError(emsg)

    sv_shape = shap_values.shape
    if len(sv_shape) == 1:
        emsg = (
            "The beeswarm plot does not support plotting a single instance, please pass "
            "an explanation matrix with many instances!"
        )
        raise ValueError(emsg)
    elif len(sv_shape) > 2:
        emsg = (
            "The beeswarm plot does not support plotting explanations with instances that have more "
            "than one dimension!"
        )
        raise ValueError(emsg)

    shap_exp = shap_values
    # we make a copy here, because later there are places that might modify this array
    values = np.copy(shap_exp.values)
    features = shap_exp.data
    if scipy.sparse.issparse(features):
        features = features.toarray()
    feature_names = shap_exp.feature_names
    # if out_names is None: # TODO: waiting for slicer support
    #     out_names = shap_exp.output_names

    order = convert_ordering(order, values)

    # multi_class = False
    # if isinstance(values, list):
    #     multi_class = True
    #     if plot_type is None:
    #         plot_type = "bar" # default for multi-output explanations
    #     assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    # else:
    #     if plot_type is None:
    #         plot_type = "dot" # default for single output explanations
    #     assert len(values.shape) != 1, "Summary plots need a matrix of values, not a vector."

    # default color:
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

    if features is not None:
        shape_msg = (
            "The shape of the shap_values matrix does not match the shape "
            "of the provided data matrix."
        )
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if log_scale:
        pl.xscale('symlog')

    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else:
            partition_tree = None
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering

    if partition_tree is not None:
        if partition_tree.shape[1] != 4:
            emsg = (
                "The clustering provided by the Explanation object does not seem to "
                "be a partition tree (which is all shap.plots.bar supports)!"
            )
            raise ValueError(emsg)

    # FIXME: introduce beeswarm interaction values as a separate function `beeswarm_interaction()` (?)
    #   In the meantime, users can use the `shap.summary_plot()` function.
    #
    # # plotting SHAP interaction values
    # if len(values.shape) == 3:
    #
    #     if plot_type == "compact_dot":
    #         new_values = values.reshape(values.shape[0], -1)
    #         new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)
    #
    #         new_feature_names = []
    #         for c1 in feature_names:
    #             for c2 in feature_names:
    #                 if c1 == c2:
    #                     new_feature_names.append(c1)
    #                 else:
    #                     new_feature_names.append(c1 + "* - " + c2)
    #
    #         return beeswarm(
    #             new_values, new_features, new_feature_names,
    #             max_display=max_display, plot_type="dot", color=color, axis_color=axis_color,
    #             title=title, alpha=alpha, show=show, sort=sort,
    #             color_bar=color_bar, plot_size=plot_size, class_names=class_names,
    #             color_bar_label="*" + color_bar_label
    #         )
    #
    #     if max_display is None:
    #         max_display = 7
    #     else:
    #         max_display = min(len(feature_names), max_display)
    #
    #     interaction_sort_inds = order#np.argsort(-np.abs(values.sum(1)).sum(0))
    #
    #     # get plotting limits
    #     delta = 1.0 / (values.shape[1] ** 2)
    #     slow = np.nanpercentile(values, delta)
    #     shigh = np.nanpercentile(values, 100 - delta)
    #     v = max(abs(slow), abs(shigh))
    #     slow = -v
    #     shigh = v
    #
    #     pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
    #     pl.subplot(1, max_display, 1)
    #     proj_values = values[:, interaction_sort_inds[0], interaction_sort_inds]
    #     proj_values[:, 1:] *= 2  # because off diag effects are split in half
    #     beeswarm(
    #         proj_values, features[:, interaction_sort_inds] if features is not None else None,
    #         feature_names=feature_names[interaction_sort_inds],
    #         sort=False, show=False, color_bar=False,
    #         plot_size=None,
    #         max_display=max_display
    #     )
    #     pl.xlim((slow, shigh))
    #     pl.xlabel("")
    #     title_length_limit = 11
    #     pl.title(shorten_text(feature_names[interaction_sort_inds[0]], title_length_limit))
    #     for i in range(1, min(len(interaction_sort_inds), max_display)):
    #         ind = interaction_sort_inds[i]
    #         pl.subplot(1, max_display, i + 1)
    #         proj_values = values[:, ind, interaction_sort_inds]
    #         proj_values *= 2
    #         proj_values[:, i] /= 2  # because only off diag effects are split in half
    #         summary(
    #             proj_values, features[:, interaction_sort_inds] if features is not None else None,
    #             sort=False,
    #             feature_names=["" for i in range(len(feature_names))],
    #             show=False,
    #             color_bar=False,
    #             plot_size=None,
    #             max_display=max_display
    #         )
    #         pl.xlim((slow, shigh))
    #         pl.xlabel("")
    #         if i == min(len(interaction_sort_inds), max_display) // 2:
    #             pl.xlabel(labels['INTERACTION_VALUE'])
    #         pl.title(shorten_text(feature_names[ind], title_length_limit))
    #     pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
    #     pl.subplots_adjust(hspace=0, wspace=0.1)
    #     if show:
    #         pl.show()
    #     return

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()
    while True:
        feature_order = convert_ordering(order, Explanation(np.abs(values)))
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the partition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= cluster_threshold:
                #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for i in range(len(values)):
                    values[:,ind1] += values[:,ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum([values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

    # build our y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]
    if num_features < len(values[0]):
        yticklabels[-1] = "%d other buses" % num_cut

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, min(len(feature_order), max_display) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
        except Exception:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if safe_isinstance(color, "matplotlib.colors.Colormap") and features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax

            if features.shape[0] != len(shaps):
                emsg = "Feature and SHAP matrices must have the same number of rows!"
                raise DimensionError(emsg)

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777",
                        s=s, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                        cmap=color, vmin=vmin, vmax=vmax, s=s,
                        c=cvals, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)
        else:

            pl.scatter(shaps, pos + ys, s=s, alpha=alpha, linewidth=0, zorder=3,
                        color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)


    # draw the color bar
    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=13)
    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_inds))
    pl.xlabel(labels['VALUE'], fontsize=13)
    if show:
        pl.show()
    else:
        return pl.gca()

def heatmap(shap_values, instance_order=Explanation.hclust(), feature_values=Explanation.abs.mean(0),
            feature_order=None, max_display=10, cmap=colors.red_white_blue, show=True,
            plot_width=8, row_height = 0.2, ax=None):
    """Create a heatmap plot of a set of SHAP values.

    This plot is designed to show the population substructure of a dataset using supervised
    clustering and a heatmap.
    Supervised clustering involves clustering data points not by their original
    feature values but by their explanations.
    By default, we cluster using :func:`shap.utils.hclust_ordering`,
    but any clustering can be used to order the samples.

    Parameters
    ----------
    shap_values : shap.Explanation
        A multi-row :class:`.Explanation` object that we want to visualize in a
        cluster ordering.

    instance_order : OpChain or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct sample ordering given as an ``numpy.ndarray``.

    feature_values : OpChain or numpy.ndarray
        A function that returns a global summary value for each input feature, or an array of such values.

    feature_order : None, OpChain, or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct input feature ordering given as an ``numpy.ndarray``.
        If ``None``, then we use ``feature_values.argsort``.

    max_display : int
        The maximum number of features to display (default is 10).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    plot_width : int, default 8
        The width of the heatmap plot.
    
    row_height : float, default 0.5
        The height of each row in the heatmap plot.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot drawn onto it.

    Examples
    --------
    See `heatmap plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html>`_.

    """
    # sort the SHAP values matrix by rows and columns
    values = shap_values.values
    if issubclass(type(feature_values), OpChain):
        feature_values = feature_values.apply(Explanation(values))
    if issubclass(type(feature_values), Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    elif issubclass(type(feature_order), OpChain):
        feature_order = feature_order.apply(Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise Exception(f"Unsupported feature_order: {str(feature_order)}!")
    xlabel = "Instances"
    instance_order = convert_ordering(instance_order, shap_values)
    # if issubclass(type(instance_order), OpChain):
    #     #xlabel += " " + instance_order.summary_string("SHAP values")
    #     instance_order = instance_order.apply(Explanation(values))
    # elif not hasattr(instance_order, "__len__"):
    #     raise Exception("Unsupported instance_order: %s!" % str(instance_order))
    # else:
    #     instance_order_ops = None

    feature_names = np.array(shap_values.feature_names)[feature_order]
    values = shap_values.values[instance_order][:,feature_order]
    feature_values = feature_values[feature_order]

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, :max_display-1]
        new_values[:, -1] = values[:, max_display-1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[:max_display-1]
        new_feature_values[-1] = feature_values[max_display-1:].sum()
        feature_names = [
            *feature_names[:max_display-1],
            f"{values.shape[1] - max_display + 1} other buses",
        ]
        values = new_values
        feature_values = new_feature_values

    # define the plot size based on how many features we are plotting
    if ax is None:
        pl.gcf().set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
        ax = pl.gca()

    # plot the matrix of SHAP values as a heat map
    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / values.shape[1],
        interpolation="nearest",
        vmin=min(vmin,-vmax),
        vmax=max(-vmin,vmax),
        cmap=cmap,
    )

    # adjust the axes ticks and spines for the heat map + f(x) line chart
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines[["left", "right"]].set_visible(True)
    ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", direction="out")

    ax.set_ylim(values.shape[1] - row_height, -3)
    heatmap_yticks_pos = np.arange(values.shape[1])
    heatmap_yticks_labels = feature_names
    ax.yaxis.set_ticks(
        [-1.5, *heatmap_yticks_pos],
        [r"$f(x)$", *heatmap_yticks_labels],
        fontsize=13,
    )
    # remove the y-tick line for the f(x) label
    ax.yaxis.get_ticklines()[0].set_visible(False)

    ax.set_xlim(-0.5, values.shape[0] - 0.5)
    ax.set_xlabel(xlabel)

    # plot the f(x) line chart above the heat map
    ax.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.sum(0)
    ax.plot(
        -fx / np.abs(fx).max() - 1.5,
        color="#000000",
        linewidth=1,
    )

    # plot the bar plot on the right spine of the heat map
    bar_container = ax.barh(
        heatmap_yticks_pos,
        (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20,
        height=0.7,
        align="center",
        color="#000000",
        left=values.shape[0] * 1.0 - 0.5,
        # color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    )
    for b in bar_container:
        b.set_clip_on(False)

    # draw the color bar
    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([min(vmin, -vmax), max(-vmin, vmax)])
    cb = pl.colorbar(
        m,
        ticks=[min(vmin, -vmax), max(-vmin, vmax)],
        ax=ax,
        aspect=80,
        fraction=0.01,
        pad=0.10,  # padding between the cb and the main axes
    )
    cb.set_label(labels["VALUE"], size=12, labelpad=-10)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    # bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    # cb.ax.set_aspect((bbox.height - 0.9) * 15)
    # cb.draw_all()

    if show:
        pl.show()

    return ax

shap.plots.waterfall = waterfall
shap.plots.beeswarm = beeswarm
shap.plots.heatmap = heatmap
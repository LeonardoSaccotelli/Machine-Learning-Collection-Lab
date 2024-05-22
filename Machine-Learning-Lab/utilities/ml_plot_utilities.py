import matplotlib.pyplot as plt


def plot_2d_feature_space_by_class(X, y, feature_names, idx_feature_x, idx_feature_y, target_names):
    """Plot in a 2D features space the class
    distribution of the samples given in input.

    Args:
        :param X: two-dimensional array or matrix containing the input features.
        :param y: Array or vector containing the class labels.
        :param feature_names: List of feature names.
        :param idx_feature_x: Index of the feature to plot on the x-axis.
        :param idx_feature_y: Index of the feature to plot on the y-axis.
        :param target_names: List of string label

    Raises:
        IndexError: If the idx_feature_x or idx_feature_y are out of range of the columns of X.
    """
    # Check feature indices validity
    if not (0 <= idx_feature_x < X.shape[1]):
        raise IndexError(f"Feature index {idx_feature_x} is out of range for the dataframe columns.")

    if not (0 <= idx_feature_y < X.shape[1]):
        raise IndexError(f"Feature index {idx_feature_y} is out of range for the dataframe columns.")

    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])
    colors = plt.get_cmap('RdYlBu', len(target_names))
    plt.figure()
    plt.scatter(X[:, idx_feature_x], X[:, idx_feature_y],
                c=y, cmap=colors)
    plt.colorbar(ticks=range(0, len(target_names)), format=formatter)
    plt.clim(-0.5, 2.5)
    plt.xlabel(feature_names[idx_feature_x])
    plt.ylabel(feature_names[idx_feature_y])


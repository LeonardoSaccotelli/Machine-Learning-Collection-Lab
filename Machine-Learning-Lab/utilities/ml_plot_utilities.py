import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_2d_feature_space_by_class(X, y, feature_names, idx_feature_x, idx_feature_y, target_names):
    """
    Plot in a 2D features space the class
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

    Description:
        The plot_2d_feature_space_by_class function visualizes the distribution of sample
        classes in a 2D feature space defined by two specified features
        (idx_feature_x and idx_feature_y). Each sample's features are represented
        on the plot, with class labels (y) determining the color coding.
        The function ensures that the chosen feature indices are within the valid range
        for the input data (X). This plot helps in understanding how classes are
        distributed across different feature combinations.
    """
    # Check feature indices validity
    if not (0 <= idx_feature_x < X.shape[1]):
        raise IndexError(f"Feature index {idx_feature_x} is out of range for the dataframe columns.")

    if not (0 <= idx_feature_y < X.shape[1]):
        raise IndexError(f"Feature index {idx_feature_y} is out of range for the dataframe columns.")

    num_classes = len(target_names)
    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])
    colors = plt.get_cmap('RdYlBu', num_classes)
    plt.figure()
    plt.scatter(X[:, idx_feature_x], X[:, idx_feature_y], c=y, cmap=colors)
    plt.colorbar(ticks=range(0, num_classes), format=formatter)
    plt.xlabel(feature_names[idx_feature_x])
    plt.ylabel(feature_names[idx_feature_y])


def plot_bar_compare_stratification(y_non_stratified, y_stratified, return_plot=True, plot_summary_table=False):
    """
    Compare the distribution of class labels between non-stratified and stratified splits of a dataset.

    Args:
        :param y_non_stratified: Array-like or Series of class labels from a non-stratified split.
        :param y_stratified: Array-like or Series of class labels from a stratified split.
        :param return_plot: Whether to return a plot (default True).
        :param plot_summary_table: Whether to print a summary table of label counts (default False).

    Returns:
        :return: If `return_plot` is True, returns a matplotlib Axes object containing the bar plot.

    Raises:
        :raises ValueError: If `y_non_stratified` and `y_stratified` have different lengths.

    Description:
        The `plot_bar_compare_stratification` function compares the distribution of class labels between
        a non-stratified and a stratified split of a dataset. It counts the number of instances in each
        class for both splits, combines the counts into a DataFrame, and optionally plots a bar chart
        and/or prints a summary table.
    """

    # Count the number of instances in each class, both for
    # stratified and non-stratified strategy
    non_stratified_labels, non_stratified_counts = np.unique(y_non_stratified, return_counts=True)
    stratified_labels, stratified_counts = np.unique(y_stratified, return_counts=True)

    # Combine counts into a DataFrame
    df_non_stratified = pd.DataFrame({
        'Split': 'Non-Stratified',
        'Label': non_stratified_labels,
        'Count': non_stratified_counts
    })

    df_stratified = pd.DataFrame({
        'Split': 'Stratified',
        'Label': stratified_labels,
        'Count': stratified_counts
    })

    # Combine both DataFrames and pivot to the desired format
    df_combined = pd.concat([df_non_stratified, df_stratified])
    df_pivot = (df_combined.pivot(index='Split', columns='Label', values='Count')
                .fillna(0)
                .reset_index())

    if return_plot:
        df_pivot.plot(x='Split', kind='bar', stacked=False,
                      title='Non-Stratified split vs Stratified split', rot=0,
                      fontsize=14)

    if plot_summary_table:
        print(df_pivot)


def plot_repeated_holdout_train_test_score(train_score, test_score, n_iter, metrics_name):
    """
    Plot the training and test score for a metric defined by the user
    over a specific number of iterations.

    Args:
        :param train_score: One-dimensional array with the training score over the 'n_iter' iterations.
        :param test_score: One-dimensional array with the test score over the 'n_iter' iterations.
        :param n_iter: Number of iterations.
        :param metrics_name: Name of the metric used to evaluate model performance
        over the 'n_iter' iterations.

    Raises:
        ValueError: If the number of iterations is less than 1.
        ValueError: If the size of the training and test score arrays do not match.
        ValueError: If the size of the training score arrays does not match with n_iter.
        ValueError: If the size of the test score arrays does not match with n_iter.

    Description:
        The plot_repeated_holdout_train_test_score function plots the training
        and test scores over multiple iterations for a specified metric.
        It ensures the input arrays are consistent with the number of iterations,
        raising errors if discrepancies are found.
        This function is useful for visualizing the performance of a model across
        repeated experiments, aiding in understanding its stability and generalization.
    """
    if n_iter <= 1:
        raise ValueError(f'n_iter = {n_iter}. Run the experiment more than {n_iter} to plot the results.')

    if len(train_score) != len(test_score):
        raise ValueError(f'Size mismatch between train_score ({train_score}) and test_score: {test_score}')

    if len(train_score) != n_iter:
        raise ValueError(f'Number of iterations must be equal to the number of training scoring.')

    if len(test_score) != n_iter:
        raise ValueError(f'Number of iterations must be equal to the number of test scoring.')

    x = np.linspace(1, n_iter, n_iter)
    font = {'size': 14}
    plt.rc('font', **font)
    plt.figure(figsize=(10, 5))
    plt.plot(x, train_score, label='Training Score', marker='o')
    plt.plot(x, test_score, label='Test Score', marker='s')
    plt.xlabel('Repetition')
    plt.ylabel(metrics_name)
    plt.title('Training and Test Scores over ' + str(n_iter) + ' Repetitions', fontsize=16)
    plt.xticks(np.arange(1, n_iter + 1, step=1))
    plt.legend(loc="best")
    plt.show()

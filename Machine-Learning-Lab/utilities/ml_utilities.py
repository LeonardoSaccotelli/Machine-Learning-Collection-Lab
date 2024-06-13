import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def print_cv_results(grid, return_n_best):
    """
    Process the results of a cross-validation tuning procedure and format them into a pandas DataFrame
    sorted by the ranking of test scores.

    Args:
        :param grid: A scikit-learn GridSearchCV or RandomizedSearchCV object that has been fitted with data and parameters.
        :param return_n_best: Number of the best parameter combinations to return based on their ranking.

    Returns:
        :return: A pandas DataFrame containing the following columns:
            - 'params': Dictionary of parameter names and values, with 'estimator__' prefix removed.
            - 'mean_train_score': Mean training score across all folds.
            - 'std_train_score': Standard deviation of training scores across all folds.
            - 'mean_val_score': Mean validation score across all folds.
            - 'std_val_score': Standard deviation of validation scores across all folds.
            - 'rank_val_score': Ranking of validation scores, sorted in ascending order.

    Raises:
        :raises TypeError: If `grid` is not an instance of GridSearchCV or RandomizedSearchCV.
        :raises ValueError: If `return_n_best` is less than 1.
        :raises ValueError: If `return_n_best` exceeds the number of available parameter combinations.

    Description:
        The `print_cv_results` function processes the results of a cross-validation grid search
        or randomized search and formats them into a pandas DataFrame for analysis and visualization.
        It sorts the results by the ranking of test scores and optionally returns the top `n` best
        parameter combinations.
    """
    if not isinstance(grid, (GridSearchCV, RandomizedSearchCV)):
        raise TypeError("The parameter 'grid' must be an instance of GridSearchCV or RandomizedSearchCV.")

    if return_n_best < 1:
        raise ValueError("The parameter 'return_n_best' must be greater than or equal to 1.")

    if return_n_best > len(grid.cv_results_['params']):
        raise ValueError("The parameter 'return_n_best' exceeds the number of available parameter combinations."
                         "\nIt should be less or equal to %d",len(grid.cv_results_['params']))

    df = pd.DataFrame(grid.cv_results_)[
        ['params', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score',
         'rank_test_score']].sort_values(by='rank_test_score')
    df['params'] = df['params'].apply(
        lambda param_dict: {k.replace('estimator__', ''): v for k, v in param_dict.items()})
    df.rename(columns={'mean_test_score': 'mean_val_score',
                       'std_test_score': 'std_val_score',
                       'rank_test_score': 'rank_val_score'}, inplace=True)
    return df.head(return_n_best).reset_index(drop=True)

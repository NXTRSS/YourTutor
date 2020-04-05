import numpy as np


def estimate_skill(test_score, interval, review_ratio, max_points=100.):
    without_review_points = test_score - max_points * review_ratio
    lower_bound, upper_bound = interval
    if without_review_points < 0:
        if lower_bound > 0:
            new_range = interval - (upper_bound - lower_bound)
            return estimate_skill(test_score, new_range, review_ratio, max_points=25.)
        return (test_score / 100) * upper_bound
    return lower_bound + (test_score / 100) * (upper_bound - lower_bound)


def estimate_skills(test_scores, review_ratio):
    interval_range = 100 / test_scores.shape[1]
    intervals = [np.array([start, start + interval_range])
                 for start in np.linspace(0, 100, num=test_scores.shape[1], endpoint=False)]
    intervals = np.array(test_scores.shape[0] * [intervals])
    result = np.zeros_like(test_scores)
    for i, row in enumerate(test_scores):
        for j, item in enumerate(row):
            result[i, j] = estimate_skill(item, intervals[i, j], review_ratio)
    return result.max(axis=1)

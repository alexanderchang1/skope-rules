"""
Testing for SkopeRules algorithm (skrules.skope_rules).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import accuracy_score

from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_not_in
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import ignore_warnings


from skrules import SkopeRules

rng = check_random_state(0)

# load the iris dataset
# and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]



def test_skope_rules():
    """Check various parameter settings."""
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
               [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid({
        "feature_names": [None, ['a', 'b']],
        "precision_min": [0.],
        "recall_min": [0.],
        "n_estimators": [1],
        "max_samples": [0.5, 4],
        "max_samples_features": [0.5, 2],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "max_depth": [2],
        "max_features": ["auto", 1, 0.1],
        "min_samples_split": [2, 0.1],
        "n_jobs": [-1, 2]})

    with ignore_warnings():
        for params in grid:
            SkopeRules(random_state=rng,
                       **params).fit(X_train, y_train).predict(X_test)

    # additional parameters:
    SkopeRules(n_estimators=50,
               max_samples=1.,
               recall_min=0.,
               precision_min=0.).fit(X_train, y_train).predict(X_test)


def test_skope_rules_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data
    y = iris.target
    y = (y != 0)

    # Test max_samples
    assert_raises(ValueError,
                  SkopeRules(max_samples=-1).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(max_samples=0.0).fit, X, y)
    assert_raises(ValueError,
                  SkopeRules(max_samples=2.0).fit, X, y)
    # explicitly setting max_samples > n_samples should result in a warning.
    assert_warns_message(UserWarning,
                         "max_samples will be set to n_samples for estimation",
                         SkopeRules(max_samples=1000).fit, X, y)
    assert_no_warnings(SkopeRules(max_samples=np.int64(2)).fit, X, y)
    assert_raises(ValueError, SkopeRules(max_samples='foobar').fit, X, y)
    assert_raises(ValueError, SkopeRules(max_samples=1.5).fit, X, y)
    assert_raises(ValueError, SkopeRules(max_depth_duplication=1.5).fit, X, y)
    assert_raises(ValueError, SkopeRules().fit(X, y).predict, X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).decision_function,
                  X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).rules_vote, X[:, 1:])
    assert_raises(ValueError, SkopeRules().fit(X, y).score_top_rules,
                  X[:, 1:])


def test_max_samples_attribute():
    X = iris.data
    y = iris.target
    y = (y != 0)

    clf = SkopeRules(max_samples=1.).fit(X, y)
    assert_equal(clf.max_samples_, X.shape[0])

    clf = SkopeRules(max_samples=500)
    assert_warns_message(UserWarning,
                         "max_samples will be set to n_samples for estimation",
                         clf.fit, X, y)
    assert_equal(clf.max_samples_, X.shape[0])

    clf = SkopeRules(max_samples=0.4).fit(X, y)
    assert_equal(clf.max_samples_, 0.4*X.shape[0])


def test_skope_rules_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1.)
    clf.fit(X, y)
    decision_func = clf.decision_function(X_test)
    rules_vote = clf.rules_vote(X_test)
    score_top_rules = clf.score_top_rules(X_test)
    pred = clf.predict(X_test)
    pred_score_top_rules = clf.predict_top_rules(X_test, 1)
    # assert detect outliers:
    assert_greater(np.min(decision_func[-2:]), np.max(decision_func[:-2]))
    assert_greater(np.min(rules_vote[-2:]), np.max(rules_vote[:-2]))
    assert_greater(np.min(score_top_rules[-2:]),
                   np.max(score_top_rules[:-2]))
    assert_array_equal(pred, 6 * [0] + 2 * [1])
    assert_array_equal(pred_score_top_rules, 6 * [0] + 2 * [1])


def test_deduplication_works():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [4, -7]]
    y = [0] * 6 + [1] * 2
    X_test = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
              [10, 5], [5, -7]]
    # Test LOF
    clf = SkopeRules(random_state=rng, max_samples=1., max_depth_duplication=3)
    clf.fit(X, y)
    decision_func = clf.decision_function(X_test)
    rules_vote = clf.rules_vote(X_test)
    score_top_rules = clf.score_top_rules(X_test)
    pred = clf.predict(X_test)
    pred_score_top_rules = clf.predict_top_rules(X_test, 1)


def test_performances():
    X, y = make_blobs(n_samples=1000, random_state=0, centers=2)

    # make labels imbalanced by remove all but 100 instances from class 1
    indexes = np.ones(X.shape[0]).astype(bool)
    ind = np.array([False] * 100 + list(((y == 1)[100:])))
    indexes[ind] = 0
    X = X[indexes]
    y = y[indexes]
    n_samples, n_features = X.shape

    clf = SkopeRules()
    # fit
    clf.fit(X, y)
    # with lists
    clf.fit(X.tolist(), y.tolist())
    y_pred = clf.predict(X)
    assert_equal(y_pred.shape, (n_samples,))
    # training set performance
    assert_greater(accuracy_score(y, y_pred), 0.83)

    # decision_function agrees with predict
    decision = -clf.decision_function(X)
    assert_equal(decision.shape, (n_samples,))
    dec_pred = (decision.ravel() < 0).astype(np.int)
    assert_array_equal(dec_pred, y_pred)


def test_similarity_tree():
    # Test that rules are well splitted
    rules = [("a <= 2 and b > 45 and c <= 3 and a > 4", (1, 1, 0)),
             ("a <= 2 and b > 45 and c <= 3 and a > 4", (1, 1, 0)),
             ("a > 2 and b > 45", (0.5, 0.3, 0)),
             ("a > 2 and b > 40", (0.5, 0.2, 0)),
             ("a <= 2 and b <= 45", (1, 1, 0)),
             ("a > 2 and c <= 3", (1, 1, 0)),
             ("b > 45", (1, 1, 0)),
             ]

    sk = SkopeRules(max_depth_duplication=2)
    rulesets = sk._find_similar_rulesets(rules)
    # Assert some couples of rules are in the same bag
    idx_bags_rules = []
    for idx_rule, r in enumerate(rules):
        idx_bags_for_rule = []
        for idx_bag, bag in enumerate(rulesets):
            if r in bag:
                idx_bags_for_rule.append(idx_bag)
        idx_bags_rules.append(idx_bags_for_rule)

    assert_equal(idx_bags_rules[0], idx_bags_rules[1])
    assert_not_equal(idx_bags_rules[0], idx_bags_rules[2])
    # Assert the best rules are kept
    final_rules = sk.deduplicate(rules)
    assert_in(rules[0], final_rules)
    assert_in(rules[2], final_rules)
    assert_not_in(rules[3], final_rules)


def test_f1_score():
    clf = SkopeRules()
    rule0 = ('a > 0', (0, 0, 0))
    rule1 = ('a > 0', (0.5, 0.5, 0))
    rule2 = ('a > 0', (0.5, 0, 0))

    assert_equal(clf.f1_score(rule0), 0)
    assert_equal(clf.f1_score(rule1), 0.5)
    assert_equal(clf.f1_score(rule2), 0)


def test_query_handling():
    """Test the query handling functionality with various edge cases."""
    # Create a simple dataset
    X = pd.DataFrame({
        '__C__0': [1, 2, 3, 4, 5],
        '__C__1': [1.5, 2.5, 3.5, 4.5, 5.5],
        '__C__2': [2.0, 3.0, 4.0, 5.0, 6.0]
    })
    y = np.array([0, 0, 1, 1, 1])

    # Initialize SkopeRules with verbose mode
    clf = SkopeRules(verbose=1)
    
    # Test different query patterns
    test_queries = [
        "__C__0 <= 3",  # Simple query
        "__C__0 <= 3 and __C__1 <= 3.5",  # AND query
        "(__C__0 <= 3) & (__C__1 <= 3.5)",  # Query with parentheses and &
        "((__C__0 <= 3) & (__C__1 <= 3.5)) & (__C__2 <= 4.0)",  # Nested query
        "__C__0 == __C__0",  # Default rule
    ]
    
    for query in test_queries:
        result = clf._safe_query(X, query)
        assert isinstance(result, pd.DataFrame), f"Query failed: {query}"
        assert len(result) > 0, f"Empty result for query: {query}"


def test_rule_generation():
    """Test rule generation with the iris dataset."""
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=[f'__C__{i}' for i in range(4)])
    y = iris.target == 0  # Binary classification
    
    # Initialize SkopeRules with verbose mode
    clf = SkopeRules(
        feature_names=iris.feature_names,
        precision_min=0.3,
        recall_min=0.1,
        n_estimators=10,
        max_depth=3,
        verbose=1
    )
    
    # Fit the classifier
    clf.fit(X, y)
    
    # Test predictions
    y_pred = clf.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y)
    
    # Test rules
    assert len(clf.rules_) > 0, "No rules generated"
    
    # Test each rule
    for rule, _ in clf.rules_:
        # Verify rule format
        assert isinstance(rule, str)
        assert "and" in rule.lower() or "<=" in rule or ">" in rule
        
        # Test rule application
        result = clf._safe_query(X, rule)
        assert isinstance(result, pd.DataFrame)


def test_edge_cases():
    """Test edge cases and potential error conditions."""
    X = pd.DataFrame({
        '__C__0': [1, 2, 3],
        '__C__1': [1.5, 2.5, 3.5]
    })
    y = np.array([0, 1, 1])
    
    clf = SkopeRules(verbose=1)
    
    # Test empty query
    result = clf._safe_query(X, "")
    assert isinstance(result, pd.DataFrame)
    
    # Test invalid query
    result = clf._safe_query(X, "invalid_column <= 3")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    
    # Test query with special characters
    result = clf._safe_query(X, "__C__0 <= 3\n and __C__1 <= 3.5")
    assert isinstance(result, pd.DataFrame)
    
    # Test query with multiple conditions
    result = clf._safe_query(X, "__C__0 <= 3 and __C__1 <= 3.5 and __C__0 > 1")
    assert isinstance(result, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__])

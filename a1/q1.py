import itertools
import pandas as pd
from itertools import combinations


# Given data
data = [
    ('T', 'T', 'T', 'T', 0.0080),
    ('T', 'T', 'T', 'F', 0.0120),
    ('T', 'T', 'F', 'T', 0.0080),
    ('T', 'T', 'F', 'F', 0.0120),
    ('T', 'F', 'T', 'T', 0.0576),
    ('T', 'F', 'T', 'F', 0.0144),
    ('T', 'F', 'F', 'T', 0.1344),
    ('T', 'F', 'F', 'F', 0.0336),
    ('F', 'T', 'T', 'T', 0.0720),
    ('F', 'T', 'T', 'F', 0.1080),
    ('F', 'T', 'F', 'T', 0.0720),
    ('F', 'T', 'F', 'F', 0.1080),
    ('F', 'F', 'T', 'T', 0.0864),
    ('F', 'F', 'T', 'F', 0.0216),
    ('F', 'F', 'F', 'T', 0.2016),
    ('F', 'F', 'F', 'F', 0.0504)
]

variables = ['A', 'B', 'C', 'D']
data_df = pd.DataFrame(data, columns=variables + ['P'])


def calculate_joint_probabilities(df):
    variables = df.columns[:-1]
    prob_column = df.columns[-1]

    results = {}

    # Calculate joint probabilities for all possible combinations
    for r in range(1, len(variables) + 1):
        for combination in combinations(variables, r):
            prob = df.groupby(list(combination))[prob_column].sum()
            results[combination] = prob

    return results


# Calculate and print joint probabilities
joint_probs = calculate_joint_probabilities(data_df)

for combination, prob in joint_probs.items():
    combination_str = "".join(combination)
    print(f"P({combination_str})")
    print(prob)
    print()


def marginal_probability(df, query, given=None):
    if given is None:
        return df[df[list(query)].eq(list(query.values())).all(axis=1)]['P'].sum()
    else:
        joint_prob = df[df[list(query) + list(given)].eq(
            list(query.values()) + list(given.values())).all(axis=1)]['P'].sum()
        given_prob = df[df[list(given)].eq(
            list(given.values())).all(axis=1)]['P'].sum()
        if given_prob == 0:
            return 0
        return joint_prob / given_prob


def check_independence(df, var1, var2, given=None):
    values = ['T', 'F']
    if given is None:
        for val1 in values:
            for val2 in values:
                p1 = marginal_probability(df, {var1: val1})
                p2 = marginal_probability(df, {var2: val2})
                joint_p = marginal_probability(df, {var1: val1, var2: val2})
                if not abs(p1 * p2 - joint_p) < 1e-5:
                    return False
        return True
    else:
        for val1 in values:
            for val2 in values:
                for given_vals in itertools.product(values, repeat=len(given)):
                    given_dict = dict(zip(given, given_vals))
                    p1 = marginal_probability(df, {var1: val1}, given_dict)
                    p2 = marginal_probability(df, {var2: val2}, given_dict)
                    joint_p = marginal_probability(
                        df, {var1: val1, var2: val2}, given_dict)
                    if not abs(p1 * p2 - joint_p) < 1e-5:
                        return False
        return True


independencies = []

# Check pairwise independencies
for var1, var2 in itertools.combinations(variables, 2):
    if check_independence(data_df, var1, var2):
        independencies.append((var1, var2, None))

# Check conditional independencies
for var1, var2 in itertools.combinations(variables, 2):
    remaining_vars = [var for var in variables if var != var1 and var != var2]
    for given_size in range(1, len(remaining_vars) + 1):
        for given_vars in itertools.combinations(remaining_vars, given_size):
            if check_independence(data_df, var1, var2, given_vars):
                independencies.append((var1, var2, given_vars))

for ind in independencies:
    if ind[2] is None:
        print(f"{ind[0]} is independent of {ind[1]}")
    else:
        print(f"{ind[0]} is independent of {ind[1]} given {ind[2]}")

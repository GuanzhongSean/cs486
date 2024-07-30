import math
import random
from typing import List, Tuple
# Those libraries should be enough for your implementation.
# Do not change the existing function signatures in this file.

random.seed(486686)


def generate_neighbor(given_items: List[Tuple[int, int]], selection: List[int]) -> List[int]:
    """
    Given a list of items to choose from and a current selection,
    returns a neighboring selection by randomly adding or removing an item.

    :param given_items: a list of items to choose from
    :param selection: a current selection
    :return: a neighboring selection
    """

    neighbor = selection[:]
    i = random.randint(0, len(given_items) - 1)
    if neighbor[i] > 0:
        neighbor[i] += random.choice([-1, 1])
    else:
        neighbor[i] += 1
    return neighbor


# You may add other helper functions here.


def knapsack_solver(capacity: int, items: List[Tuple[int, int]]) -> List[int]:
    """
    Given a list of items to choose from and a maximum capacity of a knapsack,
    returns a selection of items that maximize the total value of items in the knapsack
    through the Simulated Annealing Algorithm.

    :param capacity: the maximum capacity of the knapsack
    :param items: a list of items to choose from (weight, value)
    :return: a selection of items that maximize the total value of items in the knapsack
    """

    # Initial selection
    initial_selection = [0] * len(items)
    initial_cost = 0

    # Simulated annealing parameters
    temperature = 1000.0
    cooling_rate = 0.99
    min_temperature = 0.01

    # Your implementation
    return []

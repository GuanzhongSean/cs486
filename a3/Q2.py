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


def knapsack_solver(capacity: int, items: List[Tuple[int, int]]) -> List[int]:
    """
    Given a list of items to choose from and a maximum capacity of a knapsack,
    returns a selection of items that maximize the total value of items in the knapsack
    through the Simulated Annealing Algorithm.

    :param capacity: the maximum capacity of the knapsack
    :param items: a list of items to choose from (weight, value)
    :return: a selection of items that maximize the total value of items in the knapsack
    """

    def calculate_total_value(selection: List[int]) -> int:
        nonlocal items
        return sum(selection[i] * items[i][1] for i in range(len(items)))

    def calculate_total_weight(selection: List[int]) -> int:
        nonlocal items
        return sum(selection[i] * items[i][0] for i in range(len(items)))

    # Initial selection
    current_selection = [0] * len(items)
    current_value = 0
    best_selection = current_selection[:]
    best_value = current_value

    # Simulated annealing parameters
    T = 1000.0
    cooling_rate = 0.99
    min_T = 0.01

    while T > min_T:
        neighbor_selection = generate_neighbor(items, current_selection)
        if calculate_total_weight(neighbor_selection) <= capacity:
            neighbor_value = calculate_total_value(neighbor_selection)
            delta_value = neighbor_value - current_value

            if delta_value > 0 or math.exp(delta_value / T) > random.random():
                current_selection = neighbor_selection
                current_value = neighbor_value

            if current_value > best_value:
                best_selection = current_selection[:]
                best_value = current_value

            T *= cooling_rate

    return best_selection


if __name__ == '__main__':
    capacity = 50
    items = [(10, 60), (20, 200), (30, 120)]
    result = knapsack_solver(capacity, items)
    expected = [1, 2, 0]
    if result == expected:
        print("Pass")
    else:
        print("Incorrect")
        print(f"Expected: {expected}")
        print(f"Output: {result}")

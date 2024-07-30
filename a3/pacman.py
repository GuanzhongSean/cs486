from heapq import heappop, heappush


def parse_input(input_string):
    lines = input_string.strip().split('\n')
    pacman_pos = tuple(map(int, lines[0].split()))
    food_pos = tuple(map(int, lines[1].split()))
    rows, cols = map(int, lines[2].split())
    grid = [list(line) for line in lines[3:3 + rows]]
    return pacman_pos, food_pos, grid, rows, cols


def manhattan_distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def get_neighbors(position, grid, rows, cols):
    x, y = position
    neighbors = []
    for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:  # Up, Left, Right, Down
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '%':
            neighbors.append((nx, ny))
    return neighbors


def a_star_search(pacman_pos, food_pos, grid, rows, cols, h=manhattan_distance):
    frontier = []
    heappush(frontier, (0 + h(pacman_pos, food_pos),
                        0, pacman_pos, [pacman_pos]))  # tuple(f-value, cost, cur_pos, path)
    visited = set()

    while frontier:
        _, cost, cur_pos, path = heappop(frontier)
        visited.add(cur_pos)
        if cur_pos == food_pos:
            return path

        neighbors = get_neighbors(cur_pos, grid, rows, cols)
        for neighbor in neighbors:
            if neighbor not in visited:
                heappush(frontier, (cost + 1 + h(neighbor, food_pos),
                                    cost + 1, neighbor, path + [neighbor]))

    return []


def initialize(grid: str) -> list[list]:
    pacman_pos, food_pos, grid, rows, cols = parse_input(grid)
    path = a_star_search(pacman_pos, food_pos, grid, rows, cols)
    return [[str(x), str(y)] for x, y in path]


if __name__ == '__main__':
    grid = """3 9
5 1
7 20
%%%%%%%%%%%%%%%%%%%%
%--------------%---%
%-%%-%%-%%-%%-%%-%-%
%--------P-------%-%
%%%%%%%%%%%%%%%%%%-%
%.-----------------%
%%%%%%%%%%%%%%%%%%%%"""

    output = initialize(grid)
    expected = [['3', '9'], ['3', '10'], ['3', '11'], ['3', '12'], ['3', '13'],
                ['3', '14'], ['3', '15'], ['3', '16'], ['2', '16'], ['1', '16'],
                ['1', '17'], ['1', '18'], ['2', '18'], ['3', '18'], ['4', '18'],
                ['5', '18'], ['5', '17'], ['5', '16'], ['5', '15'], ['5', '14'],
                ['5', '13'], ['5', '12'], ['5', '11'], ['5', '10'], ['5', '9'],
                ['5', '8'], ['5', '7'], ['5', '6'], ['5', '5'], ['5', '4'],
                ['5', '3'], ['5', '2'], ['5', '1']]

    if output == expected:
        print("Pass")
    else:
        print("Incorrect")
        print(f"Expected: {expected}")
        print(f"Output: {output}")

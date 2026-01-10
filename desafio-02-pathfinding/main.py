# main.py
from grid import create_grid, print_grid
from astar import astar
from constants import START, GOAL

def find_positions(layer):
    start = goal = None
    for i, row in enumerate(layer):
        for j, cell in enumerate(row):
            if cell == START:
                start = (i, j)
            elif cell == GOAL:
                goal = (i, j)
    return start, goal

grid = create_grid()
print_grid(grid)

for idx, layer in enumerate(grid):
    start, goal = find_positions(layer)
    print(f"\nExecutando A* na Pista {idx + 1}")

    path = astar(layer, start, goal)

    if path:
        print("Caminho encontrado:", path)
    else:
        print("Nenhum caminho possível.")

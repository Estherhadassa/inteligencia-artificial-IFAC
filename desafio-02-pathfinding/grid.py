# grid.py
from constants import EMPTY, START, GOAL, TRAP, LOCK, ASSASSIN

def create_grid():
    """
    3 pistas (camadas) representadas como listas
    """
    return [
        [
            ["S", ".", ".", "#", ".", "G"],
            [".", "#", ".", "L", ".", "."],
            [".", ".", ".", ".", "A", "."],
        ],
        [
            [".", ".", "#", ".", ".", "."],
            [".", "L", ".", "#", ".", "."],
            ["S", ".", ".", ".", ".", "G"],
        ],
        [
            ["S", ".", ".", ".", "#", "."],
            [".", "#", "L", ".", ".", "."],
            [".", ".", ".", "A", ".", "G"],
        ]
    ]

def print_grid(grid):
    for layer_idx, layer in enumerate(grid):
        print(f"\nPista {layer_idx + 1}")
        for row in layer:
            print(" ".join(row))

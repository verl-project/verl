import random
from dfs_solver import solve_star_battle
from get_grid import generate_star_battle_grid


if __name__ == '__main__':
    # from grids import grids
    size = random.randint(5, 8)
    grid, star_positions = generate_star_battle_grid(size=size)
    solve_star_battle(grid)
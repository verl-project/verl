#!/usr/bin/python3.6
import itertools
from operator import mul
from functools import reduce
import sys
from time import time
from math import floor

def satisfies_r(s, row, r):
	return satisfies_l(s, list(reversed(row)), r)

def satisfies_l(s, row, l):
	if len(set(row)) < s:
		return False
	if l == 0:
		return True
	seen = 0
	curr_max = 0
	for entry in row:
		if entry > curr_max:
			curr_max = entry
			seen += 1
			if entry == s or seen > l:
				break
	return l == seen

def possible_rows(s, l, r):
	p = []
	for row in itertools.permutations(range(1, s + 1)):
		if satisfies_l(s, row, l) and satisfies_r(s, row, r):
			p.append(list(row))
	return p

def format_time(s):
	sinm = 60
	sinh = 60 * sinm
	sind = 24 * sinh
	d = h = m = 0
	r = s
	if s > sind:
		d = floor(r / sind)
		r -= d * sind
	if s > sinh:
		h = floor(r / sinh)
		r -= h * sinh
	if s > sinm:
		m = floor(r / sinm)
		r -= m * sinm
	return f'{d}:{h:02d}:{m:02d}:{r:02d}'

def print_time(done, total, s):
	eta = format_time(int((total / done - 1) * s))
	print(f'progress: {done} / {total} = {done * 100 / total:.2f}%, est. max. time left: {eta}')

def print_time_found(done, total, s):
	dhms = format_time(int(s))
	print(f'Solution found after searching {done} / {total} = {done * 100 / total:.2f}% in {dhms}')

def is_valid_solution(s, solution, t, b):
	count = 0
	for i in range(s):
		c = [row[i] for row in solution]
		if not (satisfies_l(s, c, t[i]) and satisfies_r(s, c, b[i])):
			break
		count += 1
	return count == s

def solve(s, l, r, t, b, verbose,simple_output):
    rows = [possible_rows(s, l[i], r[i]) for i in range(s)]
    sol_count = 0
    total = reduce(mul, [len(row) for row in rows], 1)
    first_time = time()
    prev_time = first_time
    curr_time = first_time
    start_freq = 10000
    time_freq = start_freq
    log = ""  # 初始化日志变量

    # 记录排列总数
    log += f"现在通过笛卡尔积遍历的方式尝试所有可能的排列组合，总共有 {total} 种可能的排列组合需要尝试。\n"
    if verbose > 0:
        print(f'{total} possible solutions')

    for solution in itertools.product(*rows):
        sol_count += 1

        if (simple_output and sol_count < 4) or not simple_output: 
            log += f"现在判断第 {sol_count} 个可能的排列组合：。\n"  # 记录当前尝试的解
		
            log += f"这是基于每行的可能排列组合生成的解。\n"
            for i, row in enumerate(solution):
                log += f"  - 第 {i + 1} 行是从该行的所有可能排列中选出的一个。\n"
                log += f"    当前这一行的所有可能排列是：{rows[i]}。\n"
                log += f"    我们选择了排列 {row} 作为第 {i + 1} 行。\n"
            log += f"逐行拼接这些排列，形成完整棋盘{solution}。\n"
        if simple_output and (sol_count == 4):
            log += f"...像这样判断下去。\n"
			
        # 输出进度信息
        if verbose > 0 and sol_count % time_freq == 0:
            curr_time = time()
            if curr_time >= prev_time + verbose:
                if time_freq == start_freq:
                    time_freq = sol_count / 10
                elapsed_time = curr_time - first_time
                progress_info = f"已尝试 {sol_count} 个解，共计 {total} 个解。\n"
                log += progress_info  # 记录进度信息
                print(progress_info)
                prev_time = curr_time

        # 检查当前解是否符合要求
        if is_valid_solution(s, solution, t, b):
            curr_time = time()
            elapsed_time = curr_time - first_time
            found_info = f"这样，就找到了一个符合要求的解，这是第 {sol_count} 个尝试。\n"
            log += found_info
            log += f"有效解为：{solution}。\n"
            if verbose > 0:
                print(found_info)
            return solution, log  # 返回解和日志
        else:
            if (simple_output and sol_count < 4) or not simple_output: 
                log += f"很遗憾，这样的组合并不满足规则，它不是有效的解，现在进行下一次尝试。\n"

    # 如果没有找到解
    log += "所有排列组合尝试完毕，未找到符合条件的解。\n"
    return ["无解"], log  # 返回无解信息和日志



def solver_main(size: int, left: list[int], right: list[int], top: list[int], bottom: list[int],simple_output):
    """
    Main function to solve the skyscraper puzzle.

    Parameters:
        size (int): The dimension of the grid (maximum 9).
        left (list[int]): Left-side views.
        right (list[int]): Right-side views.
        top (list[int]): Top-side views.
        bottom (list[int]): Bottom-side views.
    """
    # Validate size
    if size > 9:
        print('Error: size should be 9 or smaller')
        return

    # Validate the lengths of the view lists
    if len(left) != size or len(right) != size or len(top) != size or len(bottom) != size:
        print('Error: All view lists must have a length equal to the size of the grid')
        return

    # Optional verbosity for debugging
    verbose = 1
    if verbose > 0:
        print("Input Parameters:")
        print(f"Size: {size}")
        print(f"Left: {left}")
        print(f"Right: {right}")
        print(f"Top: {top}")
        print(f"Bottom: {bottom}")

    # Solve the skyscraper puzzle
    solution,log = solve(size, left, right, top, bottom, verbose,simple_output)

    return solution,log
    # Print the solution grid
    print("Solution:")
    for row in solution:
        print(row)


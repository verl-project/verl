import os
import random
import logging
import time
from pysat.solvers import Glucose3

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.get# logger(__name__)

class SlitherlinkSolver:
    """Slitherlink 谜题求解器 - 用于验证生成的谜题"""

    def __init__(self):
        self.cells = None
        self.height = None
        self.width = None
        self.cell_constraints = []
        self.loop_constraints = []
        self.solution = None

    def generate_cell_constraints(self):
        """生成单元格约束"""
        def zero(e1, e2, e3, e4):
            """所有边都不存在"""
            return [[-e1], [-e2], [-e3], [-e4]]

        def one(e1, e2, e3, e4):
            """恰好有一条边"""
            return [[-e1, -e2], [-e1, -e3], [-e1, -e4],
                    [-e2, -e3], [-e2, -e4], [-e3, -e4],
                    [e1, e2, e3, e4]]

        def two(e1, e2, e3, e4):
            """恰好有两条边"""
            return [[e2, e3, e4], [e1, e3, e4],
                    [e1, e2, e4], [e1, e2, e3],
                    [-e2, -e3, -e4], [-e1, -e3, -e4],
                    [-e1, -e2, -e4], [-e1, -e2, -e3]]

        def three(e1, e2, e3, e4):
            """恰好有三条边"""
            return [[e1, e2], [e1, e3], [e1, e4],
                    [e2, e3], [e2, e4], [e3, e4],
                    [-e1, -e2, -e3, -e4]]

        self.cell_constraints = []
        cnf_builder = [zero, one, two, three]
        cell_id = -1
        for row in range(self.height):
            for col in range(self.width):
                cell_id += 1
                cell_value = self.cells[row][col]
                if cell_value is not None:
                    assert 0 <= cell_value <= 3
                    edges = [1 + e for e in self.get_cell_edges(cell_id)]
                    clauses = cnf_builder[cell_value](*edges)
                    self.cell_constraints += clauses

    def generate_loop_constraints(self):
        """生成环路约束"""
        def two(e1, e2):
            """两条边的约束"""
            return [[-e1, e2], [e1, -e2]]

        def three(e1, e2, e3):
            """三条边的约束"""
            return [[-e1, -e2, -e3],
                    [-e1, e2, e3],
                    [e1, -e2, e3],
                    [e1, e2, -e3]]

        def four(e1, e2, e3, e4):
            """四条边的约束"""
            return [[-e1, e2, e3, e4],
                    [e1, -e2, e3, e4],
                    [e1, e2, -e3, e4],
                    [e1, e2, e3, -e4],
                    [-e1, -e2, -e3],
                    [-e1, -e2, -e4],
                    [-e1, -e3, -e4],
                    [-e2, -e3, -e4]]

        num_corners = (self.width + 1) * (self.height + 1)
        constraint_fn = [None, None, two, three, four]
        self.loop_constraints = []

        for corner_id in range(num_corners):
            edges = [1 + e for e in self.get_corner_edges(corner_id)]
            if len(edges) >= 2:
                clauses = constraint_fn[len(edges)](*edges)
                self.loop_constraints += clauses

    def get_cell_edges(self, cell_id):
        """获取单元格周围的边"""
        assert 0 <= cell_id < (self.height * self.width)
        cell_row = cell_id // self.width
        cell_col = cell_id % self.width
        num_horizontal = self.height * (self.width + 1)
        upper_edge = cell_id
        lower_edge = upper_edge + self.width
        left_edge = num_horizontal + ((cell_row * (self.width + 1)) + cell_col)
        right_edge = left_edge + 1
        return [upper_edge, lower_edge, left_edge, right_edge]

    def get_corner_edges(self, corner_id):
        """获取角点周围的边"""
        assert 0 <= corner_id < (self.width + 1) * (self.height + 1)
        col = corner_id % (self.width + 1)
        row = corner_id // (self.width + 1)
        left_edge = None
        right_edge = None
        up_edge = None
        down_edge = None
        H = self.width * (self.height + 1)
        if col < self.width:
            right_edge = (self.width * row) + col
        if col > 0:
            left_edge = (self.width * row) + col - 1
        if row > 0:
            up_edge = H + corner_id - (self.width + 1)
        if row < self.height:
            down_edge = H + corner_id
        edges = [edge
                 for edge in [left_edge, right_edge, up_edge, down_edge]
                 if edge is not None]
        return edges

    def get_adjacent_edges(self, edge_id):
        """获取与边相邻的边"""
        vert_edges = self.height * (self.width + 1)
        hori_edges = self.width * (self.height + 1)
        num_edges = vert_edges + hori_edges
        num_corners = (self.width + 1) * (self.height + 1)

        # 输出调试信息
        # print(f"edge_id: {edge_id}, num_edges: {num_edges}")

        # 确保 edge_id 在有效范围内
        if not (0 <= edge_id < num_edges):
            # logger.error(f"Invalid edge_id: {edge_id}, it must be between 0 and {num_edges - 1}.")
            return []  # 发生错误时返回空列表，避免断言失败

        corners = [corner_id
                   for corner_id in range(num_corners)
                   if edge_id in self.get_corner_edges(corner_id)]

        if len(corners) != 2:
            # logger.error(f"边 {edge_id} 没有恰好连接两个角点")
            return []

        a, b = corners
        edges_a = [edge
                   for edge in self.get_corner_edges(a)
                   if edge != edge_id]
        edges_b = [edge
                   for edge in self.get_corner_edges(b)
                   if edge != edge_id]

        return edges_a + edges_b

    def validate(self, solution):
        """验证解决方案是否形成单个闭环"""
        if not solution:
            return False

        solution = [edge - 1 for edge in solution]
        far_edges = solution[1:]
        start = [solution[0]]

        while far_edges:
            nbrs = [nbr
                    for edge in start
                    for nbr in self.get_adjacent_edges(edge)
                    if nbr in far_edges]

            if not nbrs and far_edges:
                return False

            far_edges = [edge for edge in far_edges if edge not in nbrs]
            start = nbrs

        return True

    def solve(self):
        """解决谜题并返回解决方案"""
        self.generate_cell_constraints()
        self.generate_loop_constraints()

        constraints = self.cell_constraints + self.loop_constraints
        solver = Glucose3()
        for clause in constraints:
            solver.add_clause(clause)

        if solver.solve():
            model = solver.get_model()
            test_solution = [edge for edge in model if edge > 0]
            result = self.validate(test_solution)
            if result:
                self.solution = test_solution
                return True
        return False

def generate_puzzle(rows, cols, difficulty='medium', max_attempts=100, timeout=30):
    """生成 Slitherlink 谜题"""
    fill_rates = {
        'easy': 0.35,
        'medium': 0.25,  # 降低填充率，增加解的几率
        'hard': 0.20     # 更低的填充率
    }
    fill_rate = fill_rates.get(difficulty, 0.30)

    start_time = time.time()
    attempts = 0

    while attempts < max_attempts and (time.time() - start_time) < timeout:
        attempts += 1
        # # logger.info(f"尝试生成谜题 (尝试 {attempts}/{max_attempts})...")

        puzzle = [[None for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                if random.random() < fill_rate:
                    puzzle[i][j] = random.randint(0, 3)

        solver = SlitherlinkSolver()
        solver.cells = puzzle
        solver.height = rows
        solver.width = cols

        # 调用 solve 直接判断是否有解
        if solver.solve():
            # # logger.info(f"成功生成谜题 (尝试 {attempts}/{max_attempts})")
            return puzzle
        else:
            # # logger.warning(f"谜题没有解，继续尝试...")
            pass

    # logger.error(f"无法生成有解的谜题 (尝试了 {attempts} 次)")
    return None

def batch_generate(count, output_dir, difficulty='medium', max_attempts_per_puzzle=50):
    """批量生成谜题并保存到文件"""
    os.makedirs(output_dir, exist_ok=True)

    solutions_dir = os.path.join(os.path.dirname(output_dir), "solutions")
    os.makedirs(solutions_dir, exist_ok=True)

    successful = 0
    total_attempts = 0
    solved_puzzles = 0
    start_time = time.time()

    while successful < count:
        puzzle_num = successful + 1
        # logger.info(f"生成谜题 {puzzle_num}/{count}...")

        rows = random.choice([4, 5, 6])
        cols = random.choice([4, 5, 6])

        puzzle = generate_puzzle(rows, cols, difficulty, max_attempts_per_puzzle)
        if puzzle:
            filename = f"puzzle_{rows}x{cols}_{puzzle_num:04d}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                for row in puzzle:
                    f.write(''.join([str(cell) if cell is not None else '.' for cell in row]) + '\n')

            solver = SlitherlinkSolver()
            solver.cells = puzzle
            solver.height = rows
            solver.width = cols

            if solver.solve():
                solution_filename = f"solution_{rows}x{cols}_{puzzle_num:04d}.txt"
                solution_filepath = os.path.join(solutions_dir, solution_filename)

                with open(solution_filepath, 'w') as f:
                    # 保存解决方案的边列表
                    f.write(', '.join(map(str, solver.solution)))

                # logger.info(f"解决方案已保存到 {solution_filepath}")
                solved_puzzles += 1
            else:
                # logger.warning(f"无法为谜题 {puzzle_num} 找到解决方案")
                pass

            successful += 1
            # logger.info(f"谜题已保存到 {filepath}")

        total_attempts += 1

        if successful % 10 == 0 or successful == count:
            elapsed = time.time() - start_time
            avg_time = elapsed / successful if successful > 0 else 0
            # logger.info(f"进度: {successful}/{count} 谜题 (成功率: {successful / total_attempts:.2%}, 平均时间: {avg_time:.2f}秒/谜题)")

    # logger.info(f"批量生成完成! 共生成 {successful} 个谜题, 其中 {solved_puzzles} 个谜题有解, 用时 {time.time() - start_time:.2f} 秒")

# # 批量生成并保存谜题
# output_dir = './puzzles'  # 输出文件夹
# batch_generate(count=5000, output_dir=output_dir, difficulty='medium')

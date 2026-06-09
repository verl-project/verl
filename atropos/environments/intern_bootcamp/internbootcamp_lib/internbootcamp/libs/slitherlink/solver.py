"""
Slitherlink 求解器模块

此模块从 slitherlink_generator.py 导入 SlitherlinkSolver 类，
以便其他模块可以通过 from libs.slitherlink.solver import SlitherlinkSolver 导入。
"""

from libs.slitherlink.slitherlink_generator import SlitherlinkSolver

__all__ = ['SlitherlinkSolver']

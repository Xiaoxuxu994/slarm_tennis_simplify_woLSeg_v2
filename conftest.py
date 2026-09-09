"""让 `pytest tests/...` 能 import `src` / `tools`。

没有这个文件时，pytest 的 prepend 导入模式往 sys.path 里放的是**测试文件所在的
包根**——tests/ 下没有 __init__.py，于是放进去的是 tests/utils，仓库根不在路径上，
所有 `from src... import` 全部 ModuleNotFoundError。

绕过的办法是 `python -m pytest`（它会把 CWD 放进 sys.path），但那要求每个人都记得
加 `-m`；根目录放一个 conftest.py 让两种写法都能用，成本是三行。

★ 直接 `python tests/utils/test_xxx.py` 仍然不行，那时 sys.path[0] 是测试文件自己
  的目录，conftest.py 根本不会被加载。用 pytest 跑。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

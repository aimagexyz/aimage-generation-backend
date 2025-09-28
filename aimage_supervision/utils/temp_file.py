import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from aimage_supervision.settings import logger
from contextlib import asynccontextmanager


class TempFileManager:
    """临时文件管理器，用于安全地处理临时文件和目录"""

    def __init__(self, prefix: str = 'aimage_', base_dir: Optional[str] = None):
        """
        初始化临时文件管理器

        Args:
            prefix (str): 临时文件/目录名称的前缀
            base_dir (Optional[str]): 临时文件的基础目录，如果不指定则使用系统临时目录
        """
        self.prefix = prefix
        self.base_dir = base_dir or tempfile.gettempdir()
        self._temp_items: set[str] = set()

    def create_temp_dir(self) -> str:
        """
        创建临时目录

        Returns:
            str: 临时目录的路径
        """
        temp_dir = tempfile.mkdtemp(prefix=self.prefix, dir=self.base_dir)
        self._temp_items.add(temp_dir)
        return temp_dir

    def create_temp_file(self, suffix: str = '') -> str:
        """
        创建临时文件

        Args:
            suffix (str): 文件后缀名

        Returns:
            str: 临时文件的路径
        """
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix, prefix=self.prefix, dir=self.base_dir)
        os.close(fd)  # 关闭文件描述符
        self._temp_items.add(temp_path)
        return temp_path

    def cleanup(self) -> None:
        """清理所有创建的临时文件和目录"""
        for item in self._temp_items.copy():
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                elif os.path.isfile(item):
                    os.unlink(item)
                self._temp_items.remove(item)
            except Exception as e:
                logger.error(f'清理临时项目 {item} 时出错: {str(e)}')

    def __del__(self):
        """析构函数中确保清理所有临时文件"""
        self.cleanup()


@asynccontextmanager
async def temp_file_manager(prefix: str = 'aimage_', base_dir: Optional[str] = None):
    """
    异步上下文管理器，用于安全地管理临时文件的生命周期

    Example:
        async with temp_file_manager() as manager:
            temp_dir = manager.create_temp_dir()
            # 使用临时目录...
        # 退出上下文时自动清理
    """
    manager = TempFileManager(prefix=prefix, base_dir=base_dir)
    try:
        yield manager
    finally:
        manager.cleanup()

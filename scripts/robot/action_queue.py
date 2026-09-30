import numpy as np
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class ActionQueue:
    """
    严格复现 LeRobot 官方 ActionQueue.py
    """
    def __init__(self, rtc_enabled: bool = True):
        self.queue = None  # 处理后的动作 (给机器人执行)
        self.original_queue = None  # 原始动作 (用于发给服务端做 RTC 引导)
        self.lock = Lock()
        self.last_index = 0
        self.rtc_enabled = rtc_enabled # 对应 cfg.enabled

    def get(self) -> np.ndarray:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        with self.lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray:
        """对应 LeRobot: get_left_over"""
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].copy()

    def merge(self, original_actions, processed_actions, real_delay, action_index_before_inference: int = 0):
        """对应 LeRobot: merge"""
        with self.lock:
            # 1. 严格执行延迟校验
            self._check_delays(real_delay, action_index_before_inference)

            if self.rtc_enabled:
                # 2. 严格执行替换逻辑 (RTC 模式)
                self._replace_actions_queue(original_actions, processed_actions, real_delay)
                return

            # 3. 严格执行追加逻辑 (非 RTC 模式)
            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions, processed_actions, real_delay):
        """对应 LeRobot: _replace_actions_queue"""
        # 丢弃前 real_delay 步，并克隆/拷贝
        self.original_queue = original_actions[real_delay:].copy()
        self.queue = processed_actions[real_delay:].copy()
        self.last_index = 0

    def _append_actions_queue(self, original_actions, processed_actions):
        """对应 LeRobot: _append_actions_queue"""
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            return

        self.original_queue = np.concatenate([self.original_queue[self.last_index:], original_actions], axis=0)
        self.queue = np.concatenate([self.queue[self.last_index:], processed_actions], axis=0)
        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: int = None):
        """对应 LeRobot: _check_delays"""
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            logger.warning(
                f"[ACTION_QUEUE] 索引差值不等于实际延迟. "
                f"差值: {indexes_diff}, 真实延迟: {real_delay}"
            )
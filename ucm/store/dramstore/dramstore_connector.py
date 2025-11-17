#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ucm.logger import init_logger
from ucm.store.ucmstore import Task, UcmKVStoreBase

from ucm.store.dramstore import ucmdramstore

logger = init_logger(__name__)

SUCCESS = 0
FAILURE = -1

if torch.cuda.is_available():
    device = torch.cuda
elif hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.npu
else:
    raise RuntimeError(
        "No supported accelerator found. "
        "Please ensure either CUDA or NPU is available."
    )


@dataclass
class DramTask(Task):
    task_id: int = 0
    block_ids: Optional[List[str]] = None


class UcmDramStore(UcmKVStoreBase):
    """
    Dram Connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.dram_cache: Dict[str, str] = {}  # key: block_id+offset, value: block_id
        self.role = config.get("role", "worker")
        self.store = None
        
        if self.role == "scheduler":
            self.cached_blocks = set()
        else:
            # worker侧：初始化C++ store            
            self.store = ucmdramstore.DRAMStore()
            
            # 从config获取参数，使用默认值
            capacity = int(config.get("capacity", 10737418240))  # Default 10GB
            block_size = int(config.get("io_size", 262144))  # Default 256KB
            device_id = int(config.get("device", 0))
            stream_number = int(config.get("stream_number", 32))
            timeout_ms = int(config.get("timeout_ms", 30000))
            
            param = ucmdramstore.DRAMStore.Config(
                capacity, block_size, device_id, stream_number, timeout_ms
            )
            
            ret = self.store.Setup(param)
            if ret != 0:
                msg = f"Failed to initialize ucmdramstore, errcode: {ret}."
                raise RuntimeError(msg)

    def cc_store(self) -> int:
        """
        get the underlying implementation of Store

        Returns:
            cc pointer to Store
        """
        if self.role == "worker" and self.store is not None:
            return self.store.CCStoreImpl()
        return 0

    def create(self, block_ids: List[str]) -> List[int]:
        """
        create kv cache space in storage

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            success mask
        """
        return [SUCCESS] * len(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        hit_list = [block_id in self.cached_blocks for block_id in block_ids]
        return hit_list

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        """
        load kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        if self.role != "worker" or self.store is None:
            raise RuntimeError("load method should only be called from worker side")
        
        # 准备tensor指针和大小
        dst_tensor_ptr = [t.data_ptr() for t in dst_tensor]
        dst_tensor_size = [t.numel() * t.element_size() for t in dst_tensor]
        
        # 调用C++接口
        task_id = self.store.Load(block_ids, offset, dst_tensor_ptr, dst_tensor_size)
        
        
        logger.debug(f"load block {block_ids} finished, task_id: {task_id}.")
        return DramTask(task_id=task_id)

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        """
        dump kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        block_offset_index = [f"{bid}_{off}" for bid, off in zip(block_ids, offset)]
        create_result = set(self.store.AllocBatch(block_offset_index))
        if FAILURE in create_result:
            logger.warning(f"Dump failed: memory pool full or create failed, block_ids: {block_ids}")
            return DramTask(task_id=-1, block_ids=block_ids)
        # 准备tensor指针和大小
        src_tensor_ptr = [t.data_ptr() for t in src_tensor]
        src_tensor_size = [t.numel() * t.element_size() for t in src_tensor]
        
        # 调用C++接口（C++侧会在dump时自动调用create）
        task_id = self.store.Dump(block_ids, offset, src_tensor_ptr, src_tensor_size)
        
        # 在dram_cache中保存标识符
        for i, block_id in enumerate(block_ids):
            key = block_id + "_" + str(offset[i])
            self.dram_cache[key] = block_id
        logger.debug(f"dump block {block_ids} finished, task_id: {task_id}.")
        return DramTask(task_id=task_id, block_ids=block_ids)

    def fetch_data(
        self,
        block_ids: List[str],
        offset: List[int],
        dst_addr: List[int],
        size: List[int],
    ) -> Task:
        """
        load kv cache data to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_addr: List[int]: device tensor addr ptr.
            size: List[int]: device tensor size.
        Returns:
            task(Task).
        """
        task_id = self.store.Load(block_ids, offset, dst_addr, size)
        return DramTask(task_id=task_id)

    def dump_data(
        self,
        block_ids: List[str],
        offset: List[int],
        src_addr: List[int],
        size: List[int],
    ) -> Task:
        """
        dump kv cache data from device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_addr: List[int]: device tensor addr ptr.
            size: List[int]: device tensor size.
        Returns:
            task(Task).
        """
        task_id = self.store.Dump(block_ids, offset, src_addr, size)
        return DramTask(task_id=task_id)

    def wait(self, task: DramTask) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        if task.task_id == -1:
            logger.warning("Dump failed with full memory pool or create failed")
            return FAILURE
        
        # 调用C++接口（C++侧会在wait成功后自动调用commit）
        ret = self.store.Wait(task.task_id)
        # if ret == SUCCESS and task.block_ids is not None:
        #     self.store.CommitBatch(task.block_ids, True)
        return ret

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        if is_success:
            self.cached_blocks.update(block_ids)

    def check(self, task: Task) -> int:
        """
        check if kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - finished
            others - in process.
        """
        return self.store.Check(task.task_id)

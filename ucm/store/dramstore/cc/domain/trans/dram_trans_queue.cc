/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

#include "dram_trans_queue.h"
#include <cstring>
#include <string> 

namespace UC {

Status DramTransQueue::Setup(const int32_t deviceId, TaskSet* failureSet, 
                                const MemoryPool* memPool, const size_t timeoutMs) {
    this->deviceId_ = deviceId;
    this->failureSet_ = failureSet;
    this->memPool_ = memPool;
    auto success =
        this->backend_.SetWorkerInitFn([this](auto& device) { return this->Init(device); })
            .SetWorkerFn([this](auto& shard, const auto& device) { this->Work(shard, device); })
            .SetWorkerExitFn([this](auto& device) { this->Exit(device); })
            .Run();
    return success ? Status::OK() : Status::Error();
}

void DramTransQueue::Push(std::list<Task::Shard>& shards) noexcept {
    this->backend_.Push(shards);
}

bool DramTransQueue::Init(Device& device) {
    if (this->deviceId_ < 0) { return true; }
    device = DeviceFactory::Make(this->deviceId_, 262144, 512);
    if (!device) {
        return false;
    }
    return device->Setup().Success();
}

void DramTransQueue::Exit(Device& device) {
    device.reset();
}

void DramTransQueue::Work(Task::Shard& shard, const Device& device) {
    if (this->failureSet_->Contains(shard.owner)) {
        this->Done(shard, device, true);
        return;
    }
    auto status = Status::OK();
    if (shard.type == Task::Type::DUMP) {
        status = this->D2H(shard, device);
    } else {
        status = this->H2D(shard, device);
    }
    this->Done(shard, device, status.Success());
}

Status DramTransQueue::H2D(Task::Shard& shard, const Device& device) {
    // 1. 获取 pin memory buffer (hub)
    shard.buffer = device->GetBuffer(shard.length);
    if (!shard.buffer) {
        UC_ERROR("Out of memory({}).", shard.length);
        return Status::OutOfMemory();
    }
    auto hub = shard.buffer.get();
    
    // 2. 从 memory pool 读取数据到 pin memory (hub)
    if (!this->memPool_) {
        UC_ERROR("MemoryPool is null.");
        return Status::Error();
    }
    std::string blockKey = shard.block + "_" + std::to_string(shard.offset);
    size_t blockOffset = 0;
    if (!this->memPool_->GetOffset(blockKey, &blockOffset)) {
        UC_ERROR("Block({}) not found in memory pool.", shard.block);
        return Status::Error();
    }
    auto poolStart = this->memPool_->GetStartAddr().get();
    auto poolSrc = poolStart + blockOffset;
    std::memcpy(hub, poolSrc, shard.length);
    
    // 3. 从 pin memory (hub) 传输到 GPU (shard.address)
    return device->H2DAsync((std::byte*)shard.address, (std::byte*)hub, shard.length);
}

Status DramTransQueue::D2H(Task::Shard& shard, const Device& device) {
    // 1. 获取 pin memory buffer (hub)
    shard.buffer = device->GetBuffer(shard.length);
    if (!shard.buffer) {
        UC_ERROR("Out of memory({}).", shard.length);
        return Status::OutOfMemory();
    }
    auto hub = shard.buffer.get();
    
    // 2. 从 GPU (shard.address) 传输到 pin memory (hub)
    auto status = device->D2HSync((std::byte*)hub, (std::byte*)shard.address, shard.length);
    if (status.Failure()) {
        return status;
    }
    
    // 3. 从 pin memory (hub) 写入到 memory pool
    if (!this->memPool_) {
        UC_ERROR("MemoryPool is null.");
        return Status::Error();
    }
    std::string blockKey = shard.block + "_" + std::to_string(shard.offset);
    size_t blockOffset = 0;
    if (!this->memPool_->GetOffset(blockKey, &blockOffset)) {
        UC_ERROR("Block({}) not found in memory pool.", shard.block);
        return Status::Error();
    }
    auto poolStart = this->memPool_->GetStartAddr().get();
    auto poolDst = poolStart + blockOffset;
    std::memcpy(poolDst, hub, shard.length);
    
    return Status::OK();
}

void DramTransQueue::Done(Task::Shard& shard, const Device& device, const bool success) {
    if (!success) { this->failureSet_->Insert(shard.owner); }
    if (!shard.done) { return; }
    if (device) {
        if (device->Synchronized().Failure()) { this->failureSet_->Insert(shard.owner); }
    }
    shard.done();
}

} // namespace UC
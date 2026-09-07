# UCM Encoder Cache Connector 设计

状态：Proposed  
目标版本：vLLM V1 engine 的 EC Connector + UCM Store V1；Worker 主目标为
V2 GPU model runner `worker/gpu/ec_connector.py`  
Store 由配置选择；本文以 `Cache|Posix` 为例

## 1. 背景

vLLM 的多模态 Encoder Cache（下文简称 EC）在 Worker 侧表现为：

```python
encoder_cache: dict[str, torch.Tensor]
```

其中：

- key 是多模态 item 的 `identifier`，通常等同于 vLLM 的 `mm_hash`；
- value 是一个连续的二维 tensor，逻辑形状为 `[N, D]`；
- `N` 是该 item 的 encoder embedding 数量，可由 Scheduler 侧的
  `request.get_num_encoder_embeds(index)` 获得；
- `D` 是 encoder 经过 projector/merger 后的输出宽度，在一个模型实例及其
  multimodal 配置生命周期内固定；
- value 是完整 item 的 encoder 输出，不是 vLLM KV cache block。

UCM Pipeline Store 以固定大小 block 为寻址和 I/O 单位，接口为：

```python
lookup(block_ids: list[bytes]) -> list[bool]
load(block_ids, shard_index, dst_tensor) -> Task
dump(block_ids, shard_index, src_tensor) -> Task
wait(task) -> None
check(task) -> bool
```

因此不能把一个变长的 `[N, D]` tensor 直接映射为一个 UCM block。本文采用
固定大小 chunk，将一个 EC item 映射为多个 UCM block。

## 2. 设计结论

本设计作出以下决策：

1. 只使用一个 UCM Store，由 `ucm_connector_name` 选择具体实现。
2. 不修改 UCM Store，不要求 Store 支持“identifier 到 block 链表”的索引。
3. 不在 Store 中保存 manifest，也不引入第二个元数据 Store。
4. 在服务启动期根据 `VllmConfig` 确定 `D`、dtype 和固定 chunk layout。
5. Scheduler 在 `ensure_cache_available()` 中建立请求和 identifier 的内存元数据，
   但不执行 Store lookup。
6. `has_cache_item(identifier)` 根据内存中的 identifier state 取得全部 chunk key，调用
   `store.lookup(chunk_ids)`；只有所有 chunk 都存在才算 EC 命中。
7. Worker 以一个完整 identifier 的所有 chunk 为一次逻辑 load/dump 单元。
8. `identifier_to_blocks` 的每个 entry 维护 request 级 ref count，用于在最后一个
   活跃请求不再引用 identifier 时立即删除其 state。
9. UCM GC、Cache 淘汰和 Posix 删除后，下一次 `has_cache_item()` 会重新 lookup，
   不会因为 Scheduler 内存中仍有 state 而产生假命中。

## 3. 范围

### 3.1 本期包含

- image/video encoder output 的完整 item 级复用；
- 固定 chunk layout；
- `Cache|Posix` 的本机共享内存热层与 Posix 冷层；
- Scheduler/Worker 分离；
- TP/PCP rank 间复制型 EC 的单 rank dump、全 rank load；
- 请求完成、取消、abort、preemption 和 chunked prefill 的元数据生命周期；
- 同一个 Store 中保存所有 EC chunk；
- Store 的异步 Task 接口和最小正确性同步方案。

### 3.2 本期不包含

- Store 内新增可变长对象或链表对象；
- 部分 chunk 命中时的部分 EC 复用；
- 不同模型、不同权重 revision 或不同 encoder layout 之间的 EC 共享；
- 跨模型的 tensor 转码；
- 将 manifest 持久化到 Posix；
- 对动态改变 EC 输出宽度的未知模型自动猜测；
- 对 Scheduler 预测的 `N` 与 Worker 实际输出行数不一致的模型静默兼容。

## 4. 核心不变量

实现必须满足以下不变量：

```text
layout 在服务生命周期内不可变
D_scheduler == D_worker == tensor.shape[1]
dtype_scheduler == dtype_worker == tensor.dtype
N_scheduler == tensor.shape[0]
chunk key 对相同内容和相同 layout 必须稳定
不同 layout 的 chunk key 必须隔离
只有全部 chunk lookup 命中才返回 EC hit
lookup 结果不能跨调度 step 长期缓存
```

任何 shape、dtype 或 layout 不一致必须 fail closed：本次 item 不得写入或加载为
合法 EC，不能截断、补猜或动态重建 Store。

## 5. 服务启动期确定 Encoder Cache Layout

### 5.1 初始化时机

Scheduler 和 Worker 构造 EC connector 时都能获得完整的 `VllmConfig`：

```python
UCMECConnector(vllm_config, role)
```

因此 connector 构造期即可完成：

```text
VllmConfig
  -> resolve_encoder_cache_layout()
  -> D / dtype / element_size
  -> rows_per_chunk / chunk_bytes
  -> layout_id / cache_namespace
  -> initialize configured UCM Store
```

不依赖第一次 encoder forward，也不依赖 Worker 模型对象。

### 5.2 基础宽度接口

对大多数多模态模型：

```python
D = vllm_config.model_config.get_inputs_embeds_size()
```

这是 projector 后送入语言模型的 `inputs_embeds` 宽度。它不是所有模型的完整 EC
layout API；特殊布局需要明确的结构约定或显式 EC 宽度。

### 5.3 宽度解析顺序

`resolve_encoder_cache_layout()` 集中解析宽度、dtype 和 chunk layout，不根据模型名称、
路径或 architecture 名称选择配置分支：

1. `encoder_cache_hidden_dim` 非空时，优先采用用户指定的 EC 宽度。
2. 未显式配置且 `vision_config` 同时提供 `out_hidden_size` 和
   `deepstack_visual_indexes` 时，按拼接布局计算
   `D = out_hidden_size * (1 + len(deepstack_visual_indexes))`；空索引列表的倍率为 1。
3. 其余情况使用 vLLM 的 `model_config.get_inputs_embeds_size()`。

显式宽度不再无条件与通用输入宽度比较。完整 EC 宽度可能不同于语言模型输入宽度，
特殊布局应允许显式配置。宽度必须为正，Worker dump 仍检查真实 tensor 的 shape 和 dtype。
结构字段推导仅适用于上述 deepstack 拼接约定，不能代替所有模型的 EC layout 接口。

不支持 multimodal pruning。仅在 connector 初始化时读取 vLLM 的 multimodal 配置并
拒绝该模式；不保留专用检测 helper、配置开关或追加宽度的适配逻辑。

### 5.4 Layout 数据结构

```python
@dataclass(frozen=True)
class EncoderCacheLayout:
    width: int
    dtype: torch.dtype
    rows_per_chunk: int
    chunk_bytes: int
    layout_id: bytes
```

`layout_id` 至少包含：

- `D`；
- dtype；
- `rows_per_chunk`；

`layout_id` 只描述 block 的物理解释方式，不承载模型语义。若 Store 的 unique id 已经
保证 dtype 隔离，也可以只包含 `D` 和 `rows_per_chunk`。模型标识、权重 revision 和
会改变 encoder 输出的 processor 配置应由独立的部署 namespace 隔离，而不是在每个
identifier 状态中重复保存。

不能只依赖 `ModelConfig.compute_hash()`，因为它并不保证覆盖所有影响多模态
encoder 输出的配置。

### 5.5 Chunk 尺寸

推荐使用“固定行数 chunk”：

```text
chunk shape = [rows_per_chunk, D]
chunk_bytes = rows_per_chunk * D * element_size
```

`rows_per_chunk` 直接取 `encoder_cache_config["chunk_size"]`，必须为正整数。
不按目标字节数反推行数，也不自动修改用户指定的行数：

```python
rows_per_chunk = encoder_config["chunk_size"]
chunk_bytes = rows_per_chunk * width * element_size
```

若所选后端或 I/O 模式要求字节对齐，配置的行数必须满足其约束；例如 Posix direct I/O
通常要求 `chunk_bytes % 4096 == 0`。此约束属于后端，不是 EC 通用 layout 参数。

## 6. Store 初始化

Scheduler 使用 `device_id=-1` 的 Store 进行 lookup；Worker 使用真实 device id 的
Store 执行 load/dump。两者必须使用相同的 `unique_id`、pipeline、block layout 和
持久化路径。

Store 创建时读取 `ucm_connector_name`，浅拷贝 `ucm_connector_config`，
补充运行参数后调用 `UcmConnectorFactoryV1.create_connector(name, store_config)`。

```python
store_config.update({
    "unique_id": f"{instance_id}.ec.dp{dp_rank}",
    "device_id": device_id,
    "tensor_size_list": [layout.chunk_bytes],
    "shard_size": layout.chunk_bytes,
    "block_size": layout.chunk_bytes,
    "share_buffer_enable": True,
    "local_rank_size": vllm_config.parallel_config.tensor_parallel_size,
    "use_gdr": False,
    "posix_gc_enable": (
        role == ECConnectorRole.SCHEDULER and dp_rank == 0
    ),
})
```

`local_rank_size` 与 KV connector 的 MLA 模式一致，直接取 TP size 并覆盖用户配置。
它控制各设备加载 chunk 的遍历顺序，使不同 rank 优先获取不同共享 buffer；每个 rank
仍加载完整 EC，不改变单 writer 策略。Scheduler 使用 `device_id=-1`，不执行设备 I/O。

connector 自身还要取得 Worker tensor 分配所用的 torch device，不能只把 `device_id`
交给 Store。EC connector 不重新实现平台判断：应把当前
`UCMDirectConnector.__init__` 中的 CUDA-alike/NPU torch-device 分支提取到
`ucm.integration.vllm.device` 的公共 helper，KV connector 与 EC connector 共同调用。
Scheduler 使用 `local_rank=-1`、`device_id=-1`、`device=None`；Worker 使用
`get_world_group().local_rank`，并将该 local rank 同时交给公共 helper 和 Store。
`start_load_caches()` 只能在 Worker 侧使用返回的 `torch.device` 分配目标 storage。

不要直接继承整个 `UCMDirectConnector`，因为它是 `KVConnectorBase_V1` 的实现，会引入
不适用于 EC 的 KV layout、metadata 和初始化生命周期。这里所说的“复用”是复用提取后的
公共设备方法。当前 `ucm.integration.vllm.device.create_device()` 返回 CPU affinity/NUMA
管理对象，也不能当作 `torch.device` 使用。

每个 UCM block 只有一个 shard，调用时：

```python
shard_index = [0] * len(chunk_ids)
```

`Cache` 作为共享内存热层，`Posix` 作为持久化冷层。两层由同一个 pipeline Store
管理，connector 不维护第二份 Store 或外部 manifest。

### 6.1 GPU buffer 注册策略

EC 与 KV 的使用模式不同：KV connector 可以在启动期注册固定 paged-KV buffer；EC
输出地址随 identifier 生命周期变化。EC 不传入预注册地址或大小，
Store 使用其默认空注册表，Task 携带每次调用传入的地址：

- 强制 `use_gdr=False`；用户配置该键时输出 warning；
- `load` 每次传本次 padded storage 地址；`dump` 传原 tensor view 或局部 tail 地址；
- tensor 和 view 保持存活直到 Store Task 完成。

普通 CUDA copy stream 从代码路径上支持这种动态地址模式，不要求地址来自预注册的 KV
buffer。GDR 路径虽然存在未命中预注册范围时的按需 MR 注册，但会增加注册成本和生命周期
复杂度，不作为首版配置。若后续必须开启 GDR，应改用预分配、预注册的 EC staging pool，
然后在 staging pool 与精确 `[N,D]` encoder tensor 之间复制。

落地前必须增加真实 GPU 的启动自检：用运行期 `torch.empty()` 地址完成一次
`dump -> wait -> load -> wait -> compare`，并覆盖 full chunk 和 dump tail。自检失败时
拒绝启动 EC offload，不能静默切换到未经验证的地址模式。

## 7. Chunk Key 设计

### 7.1 Key 输入

每个 chunk 的 block id 直接复用 UCM 现有 hash/block-id helper 生成，输出格式与
UCM 当前 KV connector 保持一致：

```text
chunk_id = ucm_hash_block_id(
    "ucm-ec-v1",
    cache_namespace,
    layout_id,
    identifier,
    num_encoder_embeds,
    chunk_index
)
```

`ucm_hash_block_id()` 是对 UCM 现有 helper 的薄封装；EC connector 不再自行实现
`H128/blake2b`，也不引入第二套 block-id 格式。

其中 `cache_namespace` 在 connector 启动时由模型、权重 revision 及影响 encoder
输出的配置确定，或者由部署显式配置；它和 `layout_id` 都是 connector 级常量，不进入
每个 `IdentifierState`。

必须把 `num_encoder_embeds` 放入 key。否则同一个 identifier 在异常情况下对应
不同 `N` 时，较短对象可能错误命中较长对象的 chunk 前缀。

### 7.2 Block 数量

```python
num_chunks = ceil(num_encoder_embeds / layout.rows_per_chunk)
chunk_ids = tuple(
    make_chunk_id(layout, identifier, num_encoder_embeds, i)
    for i in range(num_chunks)
)
```

不需要持久化 manifest，因为 Scheduler 在 request 中已经知道 identifier 和
`num_encoder_embeds`，并且启动期已经知道完整 layout。

## 8. Scheduler 内存状态

### 8.1 数据结构

```python
@dataclass(slots=True)
class IdentifierState:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]
    request_ref_count: int = 0


req_to_identifiers: dict[str, tuple[str, ...]]
identifier_to_blocks: dict[str, IdentifierState]

# 仅在一个 scheduler step 内有效，只记录 positive hit
step_verified_hits: set[str]

# 同步版本：build_connector_meta() 后移交给 Worker
pending_loads: dict[str, ECLoadSpec]
```

`identifier` 已经是 `identifier_to_blocks` 的 key；`num_chunks` 等于
`len(chunk_ids)`；`valid_nbytes` 可以由 `num_embeds * width * dtype.itemsize` 计算；
`layout_id` 是 connector 级常量。因此这些字段都不在每个 identifier 中重复保存。

### 8.2 为什么需要 request ref count

同一个 identifier 可能被多个并发请求引用。如果 request A 结束时直接删除
`identifier_to_blocks[identifier]`，仍在运行的 request B 随后调用
`has_cache_item(identifier)` 时就无法取得 chunk ids，从而产生不必要的 miss。

因此每个 `IdentifierState` 维护 `request_ref_count`：

- 计数对象是引用该 identifier 的不同活跃 request；
- 同一个 request 中 identifier 重复出现多次，只计一次；
- 同一个 request 因 chunked prefill 多次调用 ensure，也只计一次；
- preemption 不减计数，因为 request 尚未结束；
- finish、abort、cancel 各减一次；
- 计数变为 0 时立即删除 `identifier_to_blocks[identifier]`。

该计数只管理 Scheduler 内存中的 identifier state 生命周期，不是 UCM Store block 的 pin
count，也不表示 Store 中仍存在数据。Store residency 仍然必须由
`has_cache_item()` 的真实 lookup 判断；UCM GC 不需要更新这个 ref count。

由于 entry 在最后一个活跃 request 结束时删除，`identifier_to_blocks` 的规模自然受
Scheduler 活跃请求数和每请求媒体数约束，不需要长期 LRU state cache。

### 8.3 `ensure_cache_available()`

职责只有两个：

1. 建立 `req_to_identifiers`；
2. 为 request 的所有 mm feature 建立或校验 `identifier_to_blocks`。

它不调用 Store lookup；只在 request 首次注册时增加一次 request ref count。

```python
def ensure_cache_available(request, num_computed_tokens):
    req_id = request.request_id
    if req_id in req_to_identifiers:
        return True

    request_states: dict[str, IdentifierState] = {}

    # 第一遍：去重、校验并构造候选 state，不修改全局 ref count。
    for index, feature in enumerate(request.mm_features):
        identifier = feature.identifier
        num_embeds = request.get_num_encoder_embeds(index)

        state = request_states.get(identifier)
        if state is not None:
            if state.num_embeds != num_embeds:
                raise EncoderCacheLayoutError(...)
            continue

        state = identifier_to_blocks.get(identifier)
        if state is not None:
            if state.num_embeds != num_embeds:
                raise EncoderCacheLayoutError(...)
        else:
            state = IdentifierState(
                num_embeds=num_embeds,
                chunk_ids=make_chunk_ids(
                    identifier=identifier,
                    num_embeds=num_embeds,
                ),
            )

        request_states[identifier] = state

    # 第二遍只遍历去重后的 identifier，提交并增加 request ref count。
    unique_identifiers = tuple(request_states)
    req_to_identifiers[req_id] = unique_identifiers
    for identifier, candidate in request_states.items():
        state = identifier_to_blocks.setdefault(identifier, candidate)
        state.request_ref_count += 1

    return True
```

chunked prefill 不会生成新的 request；多次 prefill 使用同一个 `request_id`。因此上述
`req_id in req_to_identifiers` 保证幂等，不会重复更新或无限增加状态。

本期假设 request 被 Scheduler 接纳后 `mm_features` 不发生原地变化。若未来支持同一
request id 动态追加媒体，需要将 request 的 `(identifier, N)` signature 存入状态，
检测并增量更新，不能直接按 request id 跳过。

### 8.4 `has_cache_item(identifier)`

Store lookup 只发生在该方法中：

```python
def has_cache_item(identifier):
    if not is_consumer:
        return False

    state = identifier_to_blocks.get(identifier)
    if state is None:
        return False
    found = store.lookup(list(state.chunk_ids))
    hit = len(found) == len(state.chunk_ids) and all(found)
    if hit:
        step_verified_hits.add(identifier)
    return hit
```

命中语义是 exact item hit，不使用 `lookup_on_prefix()`：

```text
[true, true, true]  -> hit
[true, true, false] -> miss
[false, false]      -> miss
```

EC 不能像文本 prefix KV 一样复用前几个 chunk，因为缺失任何 encoder embedding 都
无法构造完整的 item 输出。

`step_verified_hits` 只连接同一个调度 step 内的 `has_cache_item()` 与
`update_state_after_alloc()`；必须在 `build_connector_meta()` 中清空，不能跨 step
保存，否则可能在 UCM GC 后产生旧命中。

### 8.5 `update_state_after_alloc()`

`_try_schedule_encoder_inputs()` 先调用 `has_cache_item(identifier)`，并据此把 item
放入两条路径之一：

- hit：加入 `external_load_encoder_input`，不占用 encoder compute budget；
- miss：加入 `encoder_inputs_to_schedule`，由本 step 执行 encoder。

两条路径随后都会先调用 `encoder_cache_manager.allocate()`，再调用同一个
`update_state_after_alloc(request, index)`。该接口没有显式的 load/compute 参数，所以
connector 通过本 step 的 `step_verified_hits` 恢复刚才的决定。它的职责是把外部命中的
item 加入本 step 的 Worker load metadata，不执行 Store lookup，也不执行实际 I/O。

```python
def update_state_after_alloc(request, index):
    identifier = request.mm_features[index].identifier

    if not is_consumer or identifier not in step_verified_hits:
        return
    if identifier in pending_loads:
        return

    state = identifier_to_blocks.get(identifier)
    if state is None:
        # 该 item 已走 external-load 路径，不能静默跳过。
        raise UCMEncoderCacheError(
            identifier,
            "Missing scheduler state for verified encoder cache item",
        )

    pending_loads[identifier] = ECLoadSpec(
        num_embeds=state.num_embeds,
        chunk_ids=state.chunk_ids,
    )
```

同一个 identifier 在一个 step 内只产生一份 metadata。不要在该方法中再次 lookup。
正常 Scheduler 时序下 `state is None` 不会发生；显式状态错误用于防御未来生命周期或
调用顺序调整。这里不能静默 `return`，否则 Scheduler 已经跳过 encoder compute，却不会
给 Worker 下发 load metadata，最终只会把错误延迟成缺失 encoder output。

UCM Store 不需要 Scheduler 像 CPU connector 那样预分配目标共享内存 block，因此不维护
`pending_saves`。V2 Worker wrapper 会比较 encoder forward 前后的 `encoder_cache` keys，
并对新产生的 entry 调用 `save_caches()`；legacy/V1 则在每个 encoder output 写入 cache
时立即调用。两者最终都由 `save_caches()` 根据实际 tensor 的 `N` 和 connector 级
layout 生成 chunk ids 后 dump，角色/rank 门控也由该方法自身负责。

### 8.6 `build_connector_meta()`

```python
@dataclass(frozen=True, slots=True)
class ECLoadSpec:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]


@dataclass
class UCMECConnectorMetadata(ECConnectorMetadata):
    loads: dict[str, ECLoadSpec]
```

`build_connector_meta()` 将 pending load 状态移动到 metadata，而不是复制后继续保留：

```python
meta = UCMECConnectorMetadata(
    loads=pending_loads,
)
pending_loads = {}
step_verified_hits.clear()
return meta
```

即使 `loads` 为空，connector 启用时也应返回空 metadata 对象而不是 `None`，以便
producer Worker 进入 EC wrapper，在 encoder forward 后触发 `save_caches()`。

### 8.7 请求完成、abort 和 preemption

最终完成、abort 和 cancel 都会进入 vLLM 的 request finish/free 路径：

```python
def request_finished(request):
    identifiers = req_to_identifiers.pop(request.request_id, None)
    if identifiers is None:
        # finish hook 必须幂等，避免重复 decrement。
        return False, None

    for identifier in identifiers:
        state = identifier_to_blocks.get(identifier)
        if state is None:
            continue
        assert state.request_ref_count > 0
        state.request_ref_count -= 1
        if state.request_ref_count == 0:
            identifier_to_blocks.pop(identifier, None)

    return False, None
```

这里只删除 Scheduler 的 identifier state，不删除 Store 数据或 Posix 文件，也不把 state
存在与否直接当作 Store 命中。

Preemption 不是请求完成：

- request id 不变；
- `req_to_identifiers` 保留；
- 再次进入 waiting 时 `ensure_cache_available()` 幂等返回；
- 不重复创建 request 元数据。

## 9. Worker 数据路径

### 9.0 UCM V1 的固定长度约束

当前 `UcmKVStoreBaseV1.load/dump` 和 `load_data/dump_data` 都没有逐 block 的
`valid_length` 参数，只传 block id、shard index 和 tensor/address。这里的
`shard_index` 是逻辑 shard 序号，不是 byte offset。`Cache` 按启动期配置的
`tensor_size_list` 执行设备拷贝，`Posix` 固定读写 `shard_size`，因此
`Cache|Posix` 不能在一次调用中让尾 block 少拷贝若干字节。

所以每个 EC chunk 在 UCM 看来始终是完整的 `chunk_bytes`。不能把只有有效尾行的较小
tensor 直接传给 Store，否则 Store 仍会按完整配置长度访问该地址，存在越界读取或写入
风险。UCM 中的 variable-length tensor 支持是“启动时配置多个不同的固定 tensor size”，
不是“每次调用动态指定长度”。

### 9.1 Load

Scheduler 的 `encoder_cache_manager.allocate(request, index)` 只更新容量和引用记账，不
分配 Worker GPU tensor。Worker 的 encoder cache 是
`dict[str, torch.Tensor]`，外部命中时没有 vLLM 预先提供的目标 buffer。因此 connector
必须创建最终驻留的 GPU storage，再将 tensor 写入 `encoder_cache[identifier]`。

Worker 收到 `metadata.loads` 后，对每个不在本地 `encoder_cache` 的 identifier：

1. 计算 `K=ceil(N/R)`，分配完整 padded storage `[K,R,D]`；
2. 每个 `storage[i]` 作为一个固定大小 UCM chunk 的目标；
3. 调用一次 `store.load()`；
4. 同步版本 立即 `store.wait(task)`；
5. 成功后将 `storage.view(-1,D)[:N]` 写入 `encoder_cache[identifier]`。

截断结果是 view，因此底层 padded storage 会随 encoder cache entry 一起保留，每个 item
最多多占不到一个 chunk 的显存。首版接受该代价，以删除 load tail scratch 和尾部 copy。

这会造成 Scheduler encoder budget 与 Worker 实际 GPU allocation 的可控偏差。令 `L`
为当前由外部 load 得到的活跃 encoder cache item 集合：

```text
Scheduler_accounted_bytes(L) = sum(N_i * D * element_size)
Worker_allocated_bytes(L) = sum(ceil(N_i / R) * R * D * element_size)
0 <= Worker_allocated_bytes - Scheduler_accounted_bytes
  < len(L) * chunk_bytes
```

本地 encoder compute 得到的 `[N,D]` tensor 不含这笔 padding。换言之，Scheduler 仍按
有效 embedding 数 `N` 记账，而显存 allocator 会保留完整的 `K*R` 行；部署容量规划应
按“每个外部 load 活跃 item 最多额外一个 chunk”预留余量。

同步版本 必须在 `wait()` 成功后才插入 encoder cache。若 `load/wait` 抛出异常，删除任何
临时结果并抛出明确的 EC load error；由于 Scheduler 已选择 external-load 路径、没有安排
encoder compute，首版只能 fail closed，不能在 Worker 内静默重算。

这里的 fail closed 是服务级语义：异常会从 `start_load_caches()` 传播出
`execute_model()`，当前 vLLM EngineCore busy loop 会将其视为 fatal error 并发送
engine-dead 状态。API/frontend 进程可能仍存活，但当前及后续推理请求会失败，通常需要
重启服务。同步版本 不具备 request-level fallback 或自动 encoder recompute。

后续异步 load 必须在模型读取 output 前建立 load stream/event 到 compute stream 的依赖；
若 UCM 无法暴露可供 compute stream 等待的完成事件，则该路径仍需同步等待。

### 9.2 Dump

`save_caches()` 必须自行完成角色和唯一 writer 门控，不能依赖外层 model runner：

```python
if not self.is_producer:
    return
if self.tp_rank != 0 or self.pcp_rank != 0:
    return
```

若 PP 配置可能在非首 stage 构造 connector，还应同时要求当前 worker 是持有 multimodal
encoder 的首个 PP stage。该门控放在 connector 内，保证 legacy/V1 和 V2 两条 wrapper
路径语义一致。

当 vLLM 完成 encoder 计算后：

```python
tensor = encoder_cache[identifier]
```

首先严格校验：

```python
num_embeds = tensor.shape[0]
assert tensor.ndim == 2
assert tensor.shape[1] == layout.width
assert tensor.dtype == layout.dtype
assert tensor.is_contiguous()
```

完整 chunk 可以直接使用原 tensor 的连续 view。尾 chunk 不足
`rows_per_chunk` 时，必须分配一个完整 tail buffer，将有效行复制进去并将 padding
清零。清零既保证固定长度，也避免把未初始化的设备内存写入共享内存或文件。不能让
Store 从 tensor 尾部越界读取固定 `chunk_bytes`。

```text
chunk 0       rows [0, R)
chunk 1       rows [R, 2R)
...
tail chunk    valid rows + zero padding
```

同步版本 只为 dump tail 创建一次局部临时 tensor，完整 chunk 直接引用原 tensor：

```python
num_full_chunks, tail_rows = divmod(num_embeds, layout.rows_per_chunk)
src_chunks = [
    [tensor[i * R : (i + 1) * R]]
    for i in range(num_full_chunks)
]

tail = None
if tail_rows:
    tail = torch.zeros((R, D), dtype=tensor.dtype, device=tensor.device)
    tail[:tail_rows].copy_(tensor[num_full_chunks * R :])
    src_chunks.append([tail])
```

然后调用：

```python
chunk_ids = make_chunk_ids(identifier, num_embeds)
task = store.dump(
    list(chunk_ids),
    [0] * len(chunk_ids),
    src_chunks,
)
```

尾 buffer 和原始 encoder tensor 的引用必须至少保持到 Task 完成。同步版本 同步
`wait(task)`；dump 失败只记录日志/指标，不影响当前已经完成 encoder compute 的请求。
异步版本要在 Worker inflight table 中保存 task、源 tensor 和 tail buffer。
若 tail 的 `zero_()/copy_()` 与 UCM D2H 不在同一 stream，必须在提交 dump 前建立事件
依赖；可使用 `dump_data(..., prerequisite_handle=event_handle)`，不能只依赖 Python 调用
顺序。

每个对象浪费的传输和存储上界为：

```text
padding_bytes = ((R - N % R) % R) * D * element_size
padding_bytes < chunk_bytes
```

这是“不修改 Store API、只使用一个固定布局 Cache|Posix Store”的必要代价。

### 9.3 异步任务状态

异步 load/dump 是最终目标，但不放进首个可运行版本。同步版本 仍通过 UCM Task API
提交操作，只是在接口返回前立即 `wait()`：

```python
def build_connector_worker_meta():
    return None

def update_connector_output(connector_output):
    return

def has_pending_push_work():
    return False
```

同步版本 的失败语义：

- lookup 异常或部分命中：作为 external miss；
- load/wait 失败：不插入 `encoder_cache`，抛出明确错误并终止本次执行；
- dump/wait 失败：记录日志和指标，不影响当前请求；
- 不支持 load 失败后自动 encoder recompute。

Phase 2 再把 Scheduler -> Worker load spec 扩展为携带单调 `transfer_id`，并引入：

```python
@dataclass
class InflightTransfer:
    task: Task
    transfer_id: int | None
    identifier: str
    direction: Literal["load", "save"]
    held_tensors: tuple[torch.Tensor, ...]
    output: torch.Tensor | None = None


@dataclass
class UCMECWorkerMetadata(ECConnectorWorkerMetadata):
    # 同一个 transfer_id 会由每个参与 load 的 TP/PCP rank 各报告一次，
    # 因此必须保留 list 中的重复项，不能转换成 set。
    completed_load_ids: list[int] = field(default_factory=list)
    failed_loads: list[tuple[int, str, str]] = field(default_factory=list)

    # dump 只有唯一 writer rank 报告，按 identifier 做事件并集。
    dump_succeeded_identifiers: set[str] = field(default_factory=set)
    dump_failures: dict[str, str] = field(default_factory=dict)

    def aggregate(
        self, other: ECConnectorWorkerMetadata
    ) -> ECConnectorWorkerMetadata:
        assert isinstance(other, UCMECWorkerMetadata)
        self.completed_load_ids.extend(other.completed_load_ids)
        self.failed_loads.extend(other.failed_loads)
        self.dump_succeeded_identifiers.update(
            other.dump_succeeded_identifiers
        )
        self.dump_failures.update(other.dump_failures)
        return self
```

`build_connector_worker_meta()` 轮询 `store.check(task)`，完成后释放 held tensors，并
向 Scheduler 报告完成或失败。这里聚合的是“完成事件”，不是每个 rank 的完整状态快照：

- replicated load：成功必须收到预期 TP/PCP 参与 rank 数量的同一 `transfer_id`；任一
  `failed_loads` 报告都使该 transfer 失败；
- single-rank dump：只有唯一 writer 产生事件，所以 success/failure 都做并集，不做
  跨 rank 交集；
- dump success 只用于统计和释放 inflight 引用，不能直接写入长期 hit map；后续命中仍以
  Store lookup 为准。

`update_connector_output()` 消费上述 metadata，按 `transfer_id` 更新 load completion
计数，并记录任一 rank 的失败。identifier 不足以唯一标识异步 load，因为同一对象可能在
不同 step 再次发起传输；这也是 scheduler→worker metadata 中必须携带单调
`transfer_id` 的原因。

预期报告数应复用当前 CPU EC connector 的 executor 规则：`mp` backend 下为首个 PP
stage 的 `tp_size * pcp_size`，其他 executor 若只向 Scheduler 交付一个聚合输出则为 1；
不能无条件等待所有逻辑 rank，否则会永久持有 transfer 状态。

异步能力按以下次序落地：

1. **Phase 2A：异步 dump。** `save_caches()` 提交 Task 后由 Worker inflight table 持有
   源 tensor/tail；后续 `build_connector_worker_meta()` 报告完成或失败。dump 不在模型
   消费关键路径上，适合作为第一个异步改造。
2. **Phase 2B：异步 load。** Worker 提交 load 后，必须通过 UCM/CUDA event 建立
   load stream 到模型 compute stream 的依赖；Worker metadata 报告各 rank 的完成或失败。
   仅仅不调用 `wait()`、却没有 stream dependency，是不正确的实现。
3. **Phase 3：load failure 自动重算。** 当前 vLLM EC 没有类似 KV
   `invalid_block_ids` 的失败重算协议。仅上报 `failed_loads` 不足以恢复，因为 Scheduler
   已把 item 放入 external-load 路径且没有安排 encoder compute。完整恢复需要扩展
   Scheduler/Model Runner 协议：失败 item 不进入模型消费，释放失败 allocation，强制下一
   step 作为 external miss 重新调度 encoder compute。

因此 同步版本 已包含“同步检测 load failure 并防止错误 tensor 被消费”，但不包含“load
failure 后自动重算”。后者明确属于需要修改 vLLM EC 调度协议的 Phase 3。

### 9.4 V1/V2 model runner 双轨兼容

当前 vLLM 有两条 EC worker 调用路径：

| 路径 | save 触发方式 | wrapper producer 门控 | worker metadata 回传 |
|---|---|---|---|
| V2 `worker/gpu/ec_connector.py::ActiveECConnector` | forward 前后比较 encoder cache keys，仅保存新 entry | 有 | 有 |
| legacy/V1 `worker/ec_connector_model_runner_mixin.py` | 每个 encoder output 写入 cache 时立即调用 | 无 | 当前没有 |

本设计的完整目标路径是 V2 `ActiveECConnector`。不过 `save_caches()` 仍必须同时检查
`is_producer` 和唯一 dump rank；这样即使运行到 legacy/V1 急切保存路径，consumer 和
非 writer rank 也只会 early-return，不会重复 dump。

legacy/V1 mixin 当前在 finally 中只调用 `get_finished()` 和
`clear_connector_metadata()`，没有把 `build_connector_worker_meta()` 写入
`ECConnectorOutput`。因此它可以使用同步 load/dump 的基本数据路径，但不能使用本文的
worker metadata 完成确认。若需要完整支持 legacy/V1，必须在其 finally 中补充：

```python
output.ec_connector_worker_meta = ec_connector.build_connector_worker_meta()
```

在该 vLLM 调用点补齐并通过多 worker 聚合测试之前，异步完成回报能力明确标记为
“仅 V2 支持”，不能宣称两条 runner 路径等价。

## 10. 并行策略

### 10.1 TP/PCP

同一 DP 域内，TP 和 prefill-context-parallel rank 持有相同的完整 EC tensor。因此：

- dump：仅 `tp_rank == 0 && pcp_rank == 0` 执行，该判断必须位于 connector
  `save_caches()` 内部，不能依赖任一 model runner wrapper 已经门控；
- load：每个参与 encoder/首个 PP stage 的 TP/PCP rank 都要加载到自己的设备内存；
- chunk key 不包含 TP/PCP rank；
- Store 的共享内存 Cache 可由这些 rank 共享，减少重复 Posix 读取；
- 异步 load 的完成确认必须等待所有参与 rank，才能认为本次调度 load 完成。

### 10.2 PP

只有持有 multimodal encoder 和输入 embedding 的 PP stage 参与 EC transfer。其他 PP
stage 不初始化设备侧 EC I/O，或者只创建 no-op Worker delegate。

### 10.3 DP

每个 DP 域维护独立的 Scheduler 请求状态和共享内存 Cache namespace：

```text
unique_id = {instance_id}.ec.dp{dp_rank}
```

Posix 层可以跨 DP 域共享。chunk key 默认不包含 DP rank，从而允许不同 DP replica
复用相同模型和相同媒体的持久化 EC；`cache_namespace` 保证模型和配置隔离。

## 11. 一致性与失败语义

### 11.1 部分写入

Store 可能在 dump 完成前已经能查到部分 chunk。全量 `all(found)` 保证此时仍为 miss。
只有所有 chunk 可见后，item 才成为命中。

### 11.2 GC 和淘汰

`identifier_to_blocks` 不记录 Store residency。Cache/Posix GC 后：

```text
identifier state 仍存在
  -> has_cache_item()
  -> store.lookup(all chunk ids)
  -> 至少一个 false
  -> external miss
  -> 重新执行 encoder
```

因此不需要 Store 向 connector 发送删除通知。

### 11.3 Lookup-load 竞态

`lookup()` 与后续 `load()` 不是原子 reservation。若 block 在二者之间被 GC：

- Store load 必须返回可检测失败，不能返回未初始化数据；
- connector 将该 identifier 标记为 load failure；
- 当前 vLLM EC 路径若不能安全回退到 recompute，应先要求 UCM 在已提交 load Task
  生命周期内 pin block；
- 在具备 pin/reservation 语义前，应配置 Posix GC 避免删除刚命中的对象，并将该风险
  作为部署约束。

这是不修改 Store 方案的主要一致性边界。

### 11.4 Producer/Consumer 配置不一致

Producer 和 Consumer 的 `layout_id` 或 `cache_namespace` 不一致时，它们生成不同
chunk key，因此表现为 miss，而不是加载错误 tensor。启动日志必须打印 namespace 和
layout 摘要便于排查。

### 11.5 Shape 不一致

Worker dump 时实际 tensor 的 `D` 或 dtype 与 connector layout 不一致：

- 不写任何 chunk；
- 记录 error metric；
- 输出 identifier、actual shape、dtype 和 connector layout；
- fail closed 或让请求继续使用本地 encoder 结果；
- 不动态修改 Store layout。

Worker 直接以实际 tensor 的第一维作为 `N` 生成 key。若它与 Scheduler 从 request
推导出的 `N` 不一致，后续 Scheduler 会使用另一个 `N` 生成 chunk ids，从而表现为
miss，不会把错误形状的数据命中为合法 EC。

## 12. 性能设计

### 12.1 Scheduler

- `ensure_cache_available()` 做一次 O(M) feature 遍历和一次 O(U) 去重 identifier 提交；
- chunk key 只在 identifier 首次被活跃 request 引用时生成一次；
- request 内先去重 identifier，避免重复更新；
- `store.lookup()` 一次批量查询一个 identifier 的所有 chunk；
- 命中 identifier 只写入本 step 的 `step_verified_hits`；
- `build_connector_meta()` 后立即清空，不把 hit 跨 step 缓存。

通常 M 很小，真正成本主要是 Store lookup，而不是 Python 遍历。

### 12.2 Worker

- 一个 identifier 的 chunk 尽量一次批量 load/dump；
- 使用连续 `[num_chunks, R, D]` 接收 buffer，减少分配和 descriptor 数；
- full chunk 直接使用原 tensor view；仅 tail chunk 需要 padding buffer；
- 使用 staging/tail buffer pool，避免频繁 cuda allocation；
- 同一 DP 域只允许一个 TP/PCP rank dump；
- Cache 共享内存热命中减少重复 Posix I/O。

### 12.3 Chunk 大小权衡

chunk 太小：

- block id 数量、lookup 数量和 descriptor 数增加；
- Posix 小 I/O 和元数据压力增加。

chunk 太大：

- 尾部 padding 浪费增加；
- Cache 淘汰粒度变粗；
- 单次传输延迟增加。

示例使用 `chunk_size=128` 行；应结合模型宽度、dtype 和 benchmark 调整行数。

## 13. 配置

vLLM 入口配置：

```yaml
ec_connector: UCMECConnector
ec_role: ec_both
ec_connector_module_path: ucm.integration.vllm.ucm_ec_connector
ec_connector_extra_config:
  UCM_CONFIG_FILE: /workspace/ucm_ec_config.yaml
```

EC 专用 YAML：

```yaml
ucm_ec_connector:
  ucm_connector_name: "UcmPipelineStore"

  ucm_connector_config:
    store_pipeline: "Cache|Posix"
    storage_backends: "/mnt/test"
    posix_capacity_gb: 10240
    posix_io_engine: "psync"
    posix_data_trans_concurrency: 128
    cache_buffer_capacity_gb: 32

  encoder_cache_config:
    # Number of rows per chunk along the first dimension of [N, D].
    chunk_size: 128

    # Optional: encoder_cache_hidden_dim, cache_namespace, store_unique_id.
    # Width and namespaces are inferred when these fields are omitted.
```

`ucm_connector_name` 通过 UCM factory 选择已注册的 Store，不固定为 Pipeline Store；
`Cache|Posix` 是上面的示例配置。`ucm_connector_config` 与 `encoder_cache_config`
同级。`storage_backends` 字符串原样交给 Store，connector 不分割或转换为列表。

`Config.load_ec_config()` 直接返回 YAML 的 `ucm_ec_connector` 段，不创建 KV 参数代理。
必填字段使用下标；可选的 `encoder_cache_hidden_dim`、`cache_namespace`、
`store_unique_id` 使用 `.get()`，未配置时自动推导。显式 width 优先，实际 tensor 由 Worker 校验。
仅在组装 Store 参数时浅拷贝一次 Store 配置，不深拷贝整个 YAML。

`resolve_cache_namespace()` 直接组织模型、revision、HF commit、architecture 和 multimodal
配置 hash，不再调用独立的模型 identity helper。显式 namespace 与自动 namespace 的
hash 输入结构保持不变；模型名称仅用于缓存身份隔离，不参与宽度推导分支。

Store 的 `share_buffer_enable` 强制为 `True`，`local_rank_size` 直接覆盖为 TP size，
与 KV connector 的 MLA 分支一致。`use_gdr` 强制为 `False`；用户配置中出现该键时
输出 warning。EC 不注入预注册 GPU KV buffer 地址或大小。用户通过
`posix_capacity_gb` 配置 GC 容量；省略或设为 0 时不启动 GC。与现有 KV connector
一致，内部 `posix_gc_enable` 直接覆盖为是否 DP0 Scheduler，不作为 YAML 用户开关。
Posix Store 根据该内部 owner 标志与 `posix_capacity_gb > 0` 共同决定是否启动 GC。

用户只通过 `chunk_size` 指定第一维切块行数；不提供字节目标、对齐粒度或宽度校验开关。
当前不支持 multimodal pruning，检测到模型启用该模式时拒绝初始化。

## 14. 接口到实现的映射

| vLLM EC 接口 | UCM connector 意图 |
|---|---|
| `__init__` | 解析 layout，初始化配置指定的唯一 Store |
| `ensure_cache_available` | 幂等建立 request/identifier state，不 lookup |
| `has_cache_item` | 批量 lookup 全部 chunk，只有全命中才返回 true |
| `update_state_after_alloc` | 将外部命中的 item 加入本 step load metadata |
| `build_connector_meta` | 冻结并发送本 step load metadata，清 step cache |
| `bind_connector_metadata` | Worker 绑定本 step metadata |
| `register_caches` | 当前无调用点且不需要预注册 vLLM EC dict，保持 no-op |
| `start_load_caches` | 同步版本 padded storage + 同步 wait；Phase 2B stream/event 异步 load |
| `save_caches` | producer/唯一 rank 门控；同步版本 同步 dump；Phase 2A inflight dump |
| `build_connector_worker_meta` | 同步版本 返回 `None`；Phase 2 报告异步 load/dump 完成和失败 |
| `update_connector_output` | 同步版本 no-op；Phase 2 Scheduler 消费 Worker 完成状态 |
| `get_finished` | UCM item 级路径返回 `(None, None)`，完成事件统一走 Worker metadata |
| `request_finished` | 删除 request 映射，递减 identifier ref count，0 时删除 state |
| `has_pending_push_work` | 同步 `Cache|Posix` 首版返回 `False`；异步版本需反映 scheduler 可见的未完成工作 |
| `shutdown` | 等待/取消任务，释放 Store 和 staging pool |

## 15. 端到端流程

### 15.1 外部命中

```text
Request enters waiting
  -> ensure_cache_available(request)
       -> derive N for every identifier
       -> build identifier states and chunk ids
  -> scheduler scans encoder inputs
       -> has_cache_item(identifier)
            -> store.lookup(all chunk ids) == all true
  -> allocate local vLLM encoder cache entry
  -> update_state_after_alloc()
       -> pending_loads[identifier] = ECLoadSpec(N, chunk_ids)
  -> build_connector_meta()
  -> Worker.start_load_caches()
       -> allocate padded [K,R,D] storage
       -> store.load(all chunks)
       -> 同步版本 wait
       -> encoder_cache[identifier] = storage.view(-1,D)[:N]
  -> model consumes encoder output; encoder is not recomputed
```

### 15.2 外部 miss 后保存

```text
has_cache_item(identifier)
  -> one or more chunk miss
  -> scheduler schedules encoder compute
  -> update_state_after_alloc()
       -> no load metadata is added
  -> Worker runs encoder
  -> encoder_cache[identifier] = tensor[N,D]
  -> save_caches()
       -> derive chunk ids from identifier and actual N
       -> validate layout
       -> split full chunks + pad tail
       -> store.dump(all chunks)
       -> wait or retain inflight references
  -> later request performs fresh Store lookup and can hit
```

## 16. 测试方案

### 16.1 Layout 单元测试

- 普通布局调用 vLLM 通用输入宽度接口；
- 相同 deepstack 结构在模型重命名后仍推导出相同宽度；
- deepstack 索引为空时使用单份输出宽度，缺少完整结构时使用通用接口；
- 显式 EC 宽度优先，不调用通用宽度推导；
- 非正宽度拒绝初始化；
- connector 初始化时拒绝 pruning，且不创建 Store；
- 任意正整数 chunk 行数原样保留，字节数由行数、宽度和 dtype 决定。

### 16.2 Key 和 identifier state 单元测试

- 相同 layout/identifier/N 产生相同 chunk ids；
- 不同 layout、N、chunk index 产生不同 ids；
- chunk id 固定 16 bytes；
- `num_chunks=ceil(N/R)`；
- request 内重复 identifier 只建立一个 state；
- 两个并发 request 引用相同 identifier 时 ref count 为 2；
- 同 request 重复 identifier 和重复 ensure 不重复增加 ref count。

### 16.3 Scheduler 生命周期测试

- 重复调用 ensure 幂等；
- chunked prefill 不重复增加 request 状态；
- preemption 保留 request 状态；
- finish、abort、cancel 清理 `req_to_identifiers`；
- 一个共享 identifier 的 request 结束后，其余 request 仍可查询 state；
- 最后一个 request 结束后删除 `identifier_to_blocks` entry；
- 活跃 request 的 state 仍在时发生 Store GC，`has_cache_item` 返回 miss；
- 部分 chunk 命中返回 miss；
- `step_verified_hits` 在 `build_connector_meta` 后清空。

### 16.4 Worker I/O 测试

- N 正好整除 rows_per_chunk；
- tail chunk 为 1 行和 `R-1` 行；
- dump/load 后 tensor bitwise equal；
- tail padding 不越界；
- dtype/shape 不匹配时不写 Store；
- load failure 不向 encoder cache 插入无效 tensor；
- Phase 2：异步任务完成前 tensor 引用不释放；
- 不预注册 GPU buffer，运行期新分配地址的 full/tail `dump-load-compare` 启动自检；
- `use_gdr=False` 动态地址路径通过真实 GPU 测试。

### 16.5 并行与端到端测试

- TP=2：只发生一次 dump，两个 rank 都成功 load；
- producer/consumer 与 TP/PCP rank 的 `save_caches()` 内部门控；
- V2 wrapper 的 diff-save 路径，以及 legacy/V1 mixin 的 eager-save early-return 路径；
- worker metadata 聚合保留重复 load transfer id，任一 rank failure 优先；
- single-writer dump success/failure 使用并集；
- PCP>1：所有参与 rank 得到相同 EC；
- DP=2：独立 shm Cache，可共享 Posix EC；
- 同图并发请求；
- Producer/Consumer 冷启动；
- Cache 热命中与 Posix 冷命中；
- Posix GC 与 lookup/load 竞态注入；
- 对三个目标模型分别验证实际 EC shape。

## 17. 可观测性

至少提供：

```text
ucm_ec_lookup_items_total{result=hit|miss|partial|error}
ucm_ec_lookup_chunks_total{result=hit|miss}
ucm_ec_load_items_total{result=success|failure}
ucm_ec_dump_items_total{result=success|failure}
ucm_ec_load_bytes_total
ucm_ec_dump_bytes_total
ucm_ec_lookup_latency_seconds
ucm_ec_load_latency_seconds
ucm_ec_dump_latency_seconds
ucm_ec_identifier_entries
ucm_ec_identifier_request_refs
ucm_ec_inflight_transfers
ucm_ec_layout_mismatch_total
ucm_ec_tail_padding_bytes_total
```

启动日志打印：

```text
model / revision / D / dtype / rows_per_chunk /
chunk_bytes / layout_id / cache_namespace / pipeline / dp rank / save rank
```

日志禁止输出原始图片数据；identifier 只打印短前缀。

## 18. 分阶段实现

### 同步版本：正确性优先

- 配置期 layout resolver；
- 单 Store `Cache|Posix`；
- 固定行数 chunk；
- Scheduler 两张 map；
- `has_cache_item()` 全 chunk lookup；
- Worker 同步 load/dump；
- load 只有在 `wait()` 成功后才插入 encoder cache；失败时 fail closed；
- dump 失败记录日志/指标，不影响当前请求；
- `build_connector_worker_meta()` 返回 `None`，`update_connector_output()` no-op；
- 单 rank dump；
- V2 `ActiveECConnector` 作为完整支持的 worker wrapper；
- legacy/V1 mixin 仅保证同步数据路径和 connector 内部门控；
- shape/dtype 严格校验；
- Qwen pruning 不支持。

### Phase 2A：异步 dump

- UCM Task polling；
- Worker inflight dump table 持有源 tensor 和 tail；
- `UCMECWorkerMetadata` 上报 dump completion/failure；
- scheduler-visible inflight 计数及相应的 `has_pending_push_work()` heartbeat；
- staging/tail buffer pool。

### Phase 2B：异步 load

- Scheduler -> Worker `ECLoadSpec` 增加 `transfer_id`；
- Worker inflight load table；
- load stream 到 compute stream 的 event dependency；
- TP/PCP rank completion/failure 聚合；
- 同 step 批量多个 identifier；
- lookup/prefetch 联动。

### Phase 3：Load failure 自动重算与一致性增强

- 扩展 vLLM Scheduler/Model Runner 的 EC load-failure 协议；
- 失败 item 不进入模型消费，释放失败 allocation；
- 下一 step 强制作为 external miss 调度 encoder compute；
- Store load reservation/pin；
- item 级原子 commit 或 generation；
- partial chunk 修复和后台回填；
- pruning 等动态模型布局支持。

## 19. 验收标准

同步版本 完成需要满足：

1. 三个目标模型在启动时得到正确 D；
2. Scheduler 和所有 Worker 计算出完全相同的 layout id；
3. 任意 N 的 `[N,D]` tensor 可按固定 chunk dump/load 并 bitwise 恢复；
4. 相同图片和相同模型配置能够命中；
5. 任一 chunk 丢失、被 GC 或损坏时不得报告命中；
6. chunked prefill、preemption、finish、abort 不泄漏 request 状态；
7. identifier request ref count 正确，最后一个引用结束后不残留 state；
8. TP/PCP 下只有一个 rank dump，所有需要 EC 的 rank 均正确 load；
9. 同步 load 失败时不得把目标 tensor 插入 encoder cache；
10. 同步 dump 失败不得影响已经完成 encoder compute 的当前请求；
11. 不修改 UCM Store，不维护第二 Store，不持久化 manifest。

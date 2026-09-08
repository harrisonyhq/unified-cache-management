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
   `store.lookup_on_prefix(chunk_ids)`；返回值到达最后一个 chunk 的下标才算 EC 命中。
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
  -> cache_namespace
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
2. 未显式配置且 `vision_config` 提供 `out_hidden_size` 时，以它作为基础输出宽度。
   `deepstack_visual_indexes` 缺失、为 `None` 或为空时按 0 层处理；否则按拼接布局计算
   `D = out_hidden_size * (1 + len(deepstack_visual_indexes))`。
3. 未提供 `out_hidden_size` 时使用 vLLM 的
   `model_config.get_inputs_embeds_size()`。

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
```

`EncoderCacheLayout` 只描述 block 的物理解释方式（宽度、dtype、chunk 行数、
字节数），不承载模型语义。模型标识、权重 revision 和会改变 encoder 输出的
processor 配置由独立的部署 `cache_namespace` 隔离，而不是在每个 identifier
状态中重复保存。`rows_per_chunk` 是唯一不被 `cache_namespace` 覆盖的 layout
维度，因此它直接进入 chunk key（见第 7 节），不再额外计算中间摘要。

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

`chunk_bytes` 是 EC chunk 的逻辑数据长度。Store 使用与 KV connector 相同的尺寸推导，
将它转换成满足 I/O 对齐要求的物理 `shard_size` 和 `block_size`；用户不需要配置
EC 专用的对齐参数。

## 6. Store 初始化

Scheduler 使用 `device_id=-1` 的 Store 进行 lookup；Worker 使用真实 device id 的
Store 执行 load/dump。两者必须使用相同的 `unique_id`、pipeline、block layout 和
持久化路径。

Store 创建时读取 `ucm_connector_name`，浅拷贝 `ucm_connector_config`，将字符串形式的
`storage_backends` 按 `:` 分割为列表，再补充运行参数并调用
`UcmConnectorFactoryV1.create_connector(name, store_config)`。

```python
tensor_size_list = [layout.chunk_bytes]
store_shard_size, store_block_size = _get_store_io_sizes(
    layout.chunk_bytes, layout.chunk_bytes
)
gc_block_size = _get_store_gc_block_size(
    store_config.get("store_pipeline", ""),
    tensor_size_list,
    store_shard_size,
    store_block_size,
)
```

Worker 使用逻辑 `tensor_size_list` 和物理 `store_shard_size/store_block_size`。DP0
Scheduler 是唯一 GC owner，其 `block_size` 使用 `gc_block_size`。普通 pipeline 的
`gc_block_size` 等于物理 Store block 大小；`YuanRong|Posix` 使用与 KV connector
相同的持久化对象大小推导。这样 `posix_capacity_gb` 能按实际文件大小计算容量和淘汰。

EC 强制开启共享 buffer。缺省的 `cache_buffer_capacity_gb` 与 KV connector 一致设为
128，并在创建 Store 前检查 `/dev/shm` 容量。

`local_rank_size` 与 KV connector 的 MLA 模式一致，直接取 TP size 并覆盖用户配置。
它控制各设备加载 chunk 的遍历顺序，使不同 rank 优先获取不同共享 buffer；每个 rank
仍加载完整 EC，不改变单 writer 策略。Scheduler 使用 `device_id=-1`，不执行设备 I/O。

connector 自身还要取得 Worker tensor 分配所用的 torch device，不能只把 `device_id`
交给 Store。EC connector 不重新实现平台判断：应把当前
`UCMDirectConnector.__init__` 中的 CUDA-alike/NPU torch-device 分支提取到
`ucm.integration.vllm.device` 的公共 helper，KV connector 与 EC connector 共同调用。
Scheduler 使用 `local_rank=-1`、`device_id=-1`、`device=None`；Worker 的
`local_rank` 取 `get_world_group().local_rank`，Store `device_id` 取
`get_current_device_id()`，从而支持可见设备重映射。
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
    cache_namespace,
    rows_per_chunk,
    identifier,
    num_encoder_embeds,
    chunk_index
)
```

`ucm_hash_block_id()` 是对 UCM 现有 helper 的薄封装；EC connector 不再自行实现
`H128/blake2b`，也不引入第二套 block-id 格式。

其中 `cache_namespace` 在 connector 启动时由模型、权重 revision 及影响 encoder
输出的配置确定，或者由部署显式配置；它是 connector 级常量，不进入每个
`IdentifierState`。`rows_per_chunk` 直接放进 key，使得同一 namespace 下不同 chunk
粒度的部署生成不同 key，表现为 miss 而非加载错误行范围。

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


@dataclass(slots=True)
class RequestState:
    processed_feature_count: int = 0
    identifier_states: dict[str, IdentifierState] = field(default_factory=dict)


req_to_state: dict[str, RequestState]
identifier_to_blocks: dict[str, IdentifierState]

# 仅在一个 scheduler step 内有效，只记录 positive hit
step_verified_hits: set[str]

# 同步版本：build_connector_meta() 后移交给 Worker
pending_loads: dict[str, ECLoadSpec]
```

`identifier` 已经是 `identifier_to_blocks` 的 key；`num_chunks` 等于
`len(chunk_ids)`；`valid_nbytes` 可以由 `num_embeds * width * dtype.itemsize` 计算；
`rows_per_chunk` 是 connector 级常量。因此这些字段都不在每个 identifier 中重复保存。

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

本次修正原因：当前 vLLM streaming input 可以向同一个 request/session 的
`mm_features` 追加媒体，request_id 不变。原实现仅检查请求是否登记过就直接返回，
会漏掉新增 identifier；若该 identifier 恰好由其他请求登记，当前请求也没有计入引用，
其他请求结束时可能过早释放其元数据。因此幂等判断改为“已经校验到哪个 feature”。

对应 vLLM 源码链路（行号基于当前本地版本，均位于
[scheduler.py](D:/code/vllm/vllm/v1/core/sched/scheduler.py:2455)）：

```text
add_request(request)                                      # L2455
  → 按 request_id 找到已有 session，构造 StreamingUpdate
  → 正在等待 streaming 输入：直接更新；否则放入 streaming_queue
    → _handle_stopped_request() 取出排队更新               # L2313
  → _update_request_as_session(session, update)            # L1545
    → 调整新增媒体的 mm_position.offset
    → session.mm_features.extend(update.mm_features)       # L1572
  → 后续 schedule() 的 waiting 路径                        # L959 / L981
    → _ec_transfer_pending()                              # L2277
      → ec_connector.ensure_cache_available(request, ...) # L2282
```

这里复用原 session 对象及 request_id，只追加 feature；它是 streaming **输入**续接，
不是普通的流式输出 token。因此 connector 不能仅凭 request_id 已登记就跳过新增媒体。

`req_to_state` 持久记录每个请求的处理进度；其中 `identifier_states` 保存已登记 identifier
到共享 IdentifierState 对象的引用，替换原 `req_to_identifiers`。`identifier_to_blocks`
仍维护跨请求共享状态，同一 identifier 在各请求中引用同一个对象，不另存一份行数。

循环采用统一流程：先从历史 `registered_states` 或本批次 `new_states` 取状态，均未找到
再查全局共享表；不存在则构建，存在则只做一次行数校验。`seen_in_request` 表示该
identifier 是否已在请求历史或当前批次出现，用于判断是否新增请求引用，以及冲突应报
“请求内”还是“跨请求”。本轮不需要在不同来源的行数变量之间转换。

`new_states` 只暂存本请求首次引用的状态，整批校验成功后才增加引用并写入历史登记。
无追加时仍 O(1) 返回；有追加时只遍历新增 feature，无历史表复制。状态对象复用且 chunk
IDs 只在全局尚无对应状态时生成；这些保持算法成本，实际耗时改善需运行环境测量。

处理分为两阶段：先对新增 feature 去重和校验，全部成功后才提交新增引用及处理进度。
与请求历史、当前追加批次、其他活跃请求的行数冲突都保留原异常语义；新增批次失败不会
留下部分登记，也不会修改旧计数。请求级异常隔离放到第 11.6 节的后续专项。

```python
def ensure_cache_available(
    self,
    request: Request,
    num_computed_tokens: int,
) -> bool:
    del num_computed_tokens
    req_id = request.request_id
    request_state = self.req_to_state.get(req_id)
    if request_state is None:
        request_state = RequestState()
    num_features = len(request.mm_features)
    if num_features < request_state.processed_feature_count:
        raise EncoderCacheLayoutError(
            f"Encoder features cannot be removed from request {req_id!r}."
        )
    # Streaming updates append features; previously validated items are stable.
    if num_features == request_state.processed_feature_count:
        return True

    registered_states = request_state.identifier_states
    new_states: dict[str, IdentifierState] = {}
    for index in range(request_state.processed_feature_count, num_features):
        feature = request.mm_features[index]
        identifier = feature.identifier
        # Rows in the encoder cache tensor stored under `identifier`
        # (i.e. its first dimension).
        num_embeds = int(request.get_num_encoder_embeds(index))

        # Both previous batches and this batch refer to the same state type.
        state = registered_states.get(identifier) or new_states.get(identifier)
        seen_in_request = state is not None
        if state is None:
            state = self.identifier_to_blocks.get(identifier)

        if state is None:
            state = IdentifierState(
                num_embeds=num_embeds,
                chunk_ids=make_chunk_ids(
                    identifier=identifier,
                    num_embeds=num_embeds,
                    layout=self.layout,
                    cache_namespace=self.cache_namespace,
                    hasher=self._block_hasher,
                ),
            )
        elif state.num_embeds != num_embeds:
            scope = (
                f"within request {req_id!r}"
                if seen_in_request
                else "across active requests"
            )
            raise EncoderCacheLayoutError(
                f"Identifier {identifier!r} has inconsistent row counts "
                f"{scope}: {state.num_embeds} and {num_embeds}."
            )

        if not seen_in_request:
            new_states[identifier] = state

    # Commit only after the entire appended batch has passed validation.
    for identifier, candidate in new_states.items():
        state = self.identifier_to_blocks.setdefault(identifier, candidate)
        state.request_ref_count += 1
        registered_states[identifier] = state
    request_state.processed_feature_count = num_features
    self.req_to_state[req_id] = request_state
    return True
```

例如请求最初为 `[X, X]`，X 只增加一次引用；追加 `[X, Y, Y]` 后 X 不变、Y 增加一次；
再次 ensure 无新增 feature 时不再遍历，finish 时对 X、Y 各释放一次。

`num_computed_tokens` 继续不参与元数据登记，本 step 实际使用的 feature 由 Scheduler
选择。本实现依赖 append-only 契约：历史 feature 已校验后不变；列表缩短抛出布局异常，
等长替换或原地修改历史 feature 不在长度快速路径的支持范围，若上游允许此类更新需增加
显式版本或更新通知。最初没有媒体的请求，之后追加媒体仍可正常登记。

### 8.4 `has_cache_item(identifier)`

Store lookup 只发生在该方法中：

```python
def has_cache_item(identifier):
    if not is_consumer:
        return False

    state = identifier_to_blocks.get(identifier)
    if state is None:
        return False
    last = store.lookup_on_prefix(list(state.chunk_ids))
    hit = last == len(state.chunk_ids) - 1
    if hit:
        step_verified_hits.add(identifier)
    return hit
```

命中语义是 exact item hit。`lookup_on_prefix()` 在这里只作为“全部 chunk 命中”的
谓词使用：返回值等于最后一个 chunk 的下标（`len - 1`）等价于 `lookup()` 返回
全 true；部分前缀命中（返回值落在 `-1` 到 `len - 2` 之间）一律判 miss，不做
前缀复用：

```text
chunks [T, T, T] -> lookup_on_prefix 返回 2 == len-1 -> hit
chunks [T, T, F] -> lookup_on_prefix 返回 1          -> miss
chunks [F, F]    -> lookup_on_prefix 返回 -1         -> miss
```

EC 不能像文本 prefix KV 一样复用前几个 chunk，因为缺失任何 encoder embedding 都
无法构造完整的 item 输出。

选用 `lookup_on_prefix()` 而非 `lookup()` 是纯性能考量，命中/miss 语义完全等价：
`lookup()` 会把逐 chunk 的 bool 数组回传到 Python 侧再做聚合，`lookup_on_prefix()`
只回传一个 int，Python 侧只需一次下标比较；C++ 侧两条路径成本相同——Posix 的
`Lookup()` 本身就由 `LookupOnPrefix()` 实现（先并行扫描定位第一个缺失，再回填
前缀），Cache 层两个 `*Fast` 路径都先做同一遍 buffer `Exist()`。

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

本次修正原因：Scheduler 可能撤销本 step 已选择的请求，例如 priority 抢占路径。
原实现忽略 scheduler_output，将全部 pending_loads 发给 Worker，导致已撤销请求仍触发
无效加载。因此提交 metadata 前，按最终 num_scheduled_tokens 中 token 数大于 0 的请求，
汇总它们登记的 identifier，只保留仍有最终请求引用的 pending load。

不能只按最初触发加载的请求判断：另一请求可能通过 encoder_cache_manager 的本地命中
共享该待加载 item，没有自己的 update_state_after_alloc 回调。原请求被撤销、共享请求
仍被调度时，必须保留 load。使用最终请求的 identifier 并集覆盖这种情况。

```python
def build_connector_meta(
    self,
    scheduler_output: SchedulerOutput,
) -> UCMECConnectorMetadata:
    # A surviving request may share a pending item through the local cache
    # manager without an allocation callback of its own.
    needed_identifiers: set[str] = set()
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        if num_tokens > 0 and (state := self.req_to_state.get(req_id)) is not None:
            needed_identifiers.update(state.identifier_states)
    metadata = UCMECConnectorMetadata(
        loads={
            identifier: spec
            for identifier, spec in self.pending_loads.items()
            if identifier in needed_identifiers
        }
    )
    self.pending_loads = {}
    self.step_verified_hits.clear()
    return metadata
```

metadata.loads 使用新的字典，step 查询状态和 pending_loads 随后清空；不会修改已交给
Worker 的 metadata。过滤加载计划不减少被抢占请求的长期 identifier 引用。

这是保守的请求级过滤：最终请求包含某 identifier，即使它不在该请求本步 token 窗口内，
仍可保留对应 load。它消除无最终请求引用的加载，不承诺 feature/token 窗口级最小 I/O；
精确过滤需要 Scheduler 提供最终实际 encoder 需求。

即使 loads 为空也返回 metadata 对象，使 producer Worker 能进入 EC wrapper 并保存输出。

### 8.7 请求完成、abort 和 preemption

最终完成、abort 和 cancel 都会进入 vLLM 的 request finish/free 路径：

```python
def request_finished(
    self,
    request: Request,
) -> tuple[bool, dict[str, Any] | None]:
    request_state = self.req_to_state.pop(request.request_id, None)
    if request_state is None:
        return False, None

    for identifier in request_state.identifier_states:
        state = self.identifier_to_blocks.get(identifier)
        if state is None:
            continue
        state.request_ref_count -= 1
        if state.request_ref_count <= 0:
            del self.identifier_to_blocks[identifier]
    return False, None
```

这里只删除 Scheduler 的 identifier state，不删除 Store 数据或 Posix 文件，也不把 state
存在与否直接当作 Store 命中。

Preemption 不是请求完成：

- request id 不变；
- `req_to_state` 保留；
- 再次进入 waiting 时，无新增 feature 则幂等返回，有追加则增量登记；
- 不重复创建 request 元数据。

shutdown 同时清空 req_to_state、identifier_to_blocks 和 step 状态。

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

Store 可能在 dump 完成前已经能查到部分 chunk。`lookup_on_prefix()` 的返回值只有
在所有 chunk 可见时才会到达最后一个 chunk 的下标，部分可见仍判 miss；只有所有
chunk 可见后，item 才成为命中。

### 11.2 GC 和淘汰

`identifier_to_blocks` 不记录 Store residency。Cache/Posix GC 后：

```text
identifier state 仍存在
  -> has_cache_item()
  -> store.lookup_on_prefix(all chunk ids)
  -> 返回值停在第一个缺失 chunk 之前
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

Producer 和 Consumer 的 `rows_per_chunk` 或 `cache_namespace` 不一致时，它们生成不同
chunk key，因此表现为 miss，而不是加载错误 tensor。启动日志必须打印 namespace 和
layout 各字段便于排查。

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

### 11.6 后续专项：Load failure 与请求错误隔离（尚未实施）

本次只实施第 8 节的 streaming 增量登记和最终加载过滤。以下处理放到下一专项，
现有请求校验异常、has_cache_item 查询证据和 Worker load/save 失败行为尚未改变。

**请求布局错误隔离。** 当前 ensure 中的 EncoderCacheLayoutError 没有请求级捕获，
在本地 vLLM 可传播至 EngineCore fatal error/shutdown。目标是让可归因到请求的错误
只结束相关请求：同请求行数冲突失败当前请求；与共享状态冲突时拒绝当前冲突请求，
不覆盖既有 state，也不连带失败其他请求。streaming 新增批次不提交，旧引用等 finish
时统一释放。不能简单降级为 Store miss 后继续计算，本地 encoder cache 也按 identifier
共享，绕过 Store 无法消除 shape 歧义。全局配置无效仍应启动失败。

优先评估复用本地 vLLM 的 take_unavailable_requests → finish_requests(FINISHED_ERROR)：

- failed_requests 保存失败标记和原因直到 finish；pending_failed_req_ids 保存待上报事件。
- ensure 发现专用请求校验错误后登记失败并返回 False，不能只返回 False 导致永久等待。
- take 仅取走待上报事件，保留失败标记，避免完成清理前重新登记。
- request_finished 先清理失败状态，再处理“没有请求登记”的提前返回；有旧引用则各减一次。
- 不宽泛捕获所有 Exception；输入冲突、内部不变量损坏和 I/O 错误分别定义策略。

**Store/Worker load failure。** 先确认 Store 的失败保证和 vLLM 错误回传能力，再确定协议：

- lookup 与 load 之间 reservation/pin 的范围；缺失 chunk 和 load/wait 失败必须可检测。
- Worker 按 identifier 失败后，如何找到真正依赖它的请求，以及 TP/PP/PCP 多 rank 如何聚合。
- 已跳过 encoder compute 后，是否还保留重算输入、在哪个阶段可以安全回退。
- 无法重算时如何返回请求错误，避免从 start_load_caches 抛出并拖垮引擎。
- 异步在途 batch、abort、抢占和失败回传交错时，何时可以释放资源。

**查询证据边界。** 当前 step_verified_hits 仅添加，hit 后 miss 不撤销。专项需处理旧命中
使计算路径产生 load 的风险，但撤销查询证据不能误删其他请求已经选择的 pending load。
若 Scheduler 允许查询与分配交错，需要显式请求/feature 级决策。元数据引用计数和本次
最终加载过滤都不能代替 Store pin，也不能解决 lookup-load 淘汰竞态。

**错误返回与验证。** 首版可沿用 FINISHED_ERROR、日志记录 request_id/identifier/阶段/
错误码及行数；现有接口仅传 ID，不会自动把详细原因传给客户端。结构化客户端错误需
另行扩展；布局冲突不应描述为原样重试即可恢复。指标不使用 request_id/identifier 标签。
验收覆盖：只有错误请求且零 scheduled tokens 也能结束；混合请求中健康请求继续；
错误事件 take 后到 finish 前重复调度不重复计数；streaming/abort 不泄漏引用；部分 chunk
与多 rank 失败不使用损坏 tensor；可重算与不可重算路径分别验证。目标 vLLM 版本需复核
普通 step、batch queue 的消费时机；不能仅凭 connector 方法测试宣称服务级隔离完成。

## 12. 性能设计

### 12.1 Scheduler

- `ensure_cache_available()` 首次遍历 M 个 feature；后续仅遍历追加的 ΔM，提交新增唯一 identifier；
- 无追加时 O(1) 返回；chunk hash 成本与本次首次构建的 chunk 数有关；
- chunk key 只在 identifier 首次被活跃 request 引用时生成一次；
- request 内先去重 identifier，避免重复更新；
- `store.lookup_on_prefix()` 一次批量查询一个 identifier 的所有 chunk，返回连续
  命中前缀的末尾下标；
- 命中 identifier 只写入本 step 的 `step_verified_hits`；
- `build_connector_meta()` 汇总最终请求的 identifier 并过滤 pending loads，随后清空 step 状态。

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
同级。YAML 中 `storage_backends` 保持冒号分隔的字符串，创建 Store 前按与 KV
connector 相同的规则分割为路径列表。

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
GC 使用 Store 实际持久化对象的 `gc_block_size`，而不是未经对齐的元素数量；EC 的逻辑
chunk 字节数为 `chunk_size * width * dtype.itemsize`。

用户只通过 `chunk_size` 指定第一维切块行数；不提供字节目标、对齐粒度或宽度校验开关。
当前不支持 multimodal pruning，检测到模型启用该模式时拒绝初始化。

## 14. 接口到实现的映射

| vLLM EC 接口 | UCM connector 意图 |
|---|---|
| `__init__` | 解析 layout，初始化配置指定的唯一 Store |
| `ensure_cache_available` | 幂等建立 request/identifier state，不 lookup |
| `has_cache_item` | `lookup_on_prefix` 全 chunk 命中谓词（返回值 == len-1） |
| `update_state_after_alloc` | 将外部命中的 item 加入本 step load metadata |
| `build_connector_meta` | 按最终已调度请求过滤 load metadata，清 step cache |
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
            -> store.lookup_on_prefix(all chunk ids) == len-1
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

EC 配置、layout 和 Scheduler 状态测试统一放在 `test/test_ucm_ec_config.py`。
本次新增 11 个状态用例：追加登记/去重/跨请求共享、无追加快速返回、空请求追加、
三类行数冲突不部分提交、列表缩短、最终调度过滤、无 alloc 回调的共享请求保留 load、
空调度、finish/shutdown 和跨 step 清理。
这些用例已在实际 connector 源码与依赖替身环境中通过；本机标准导入缺少 wrapt，
也未安装 torch/vLLM，真实 Scheduler/Worker/Store 集成结果待验证。


### 16.1 Layout 单元测试

- 普通布局调用 vLLM 通用输入宽度接口；
- 相同 deepstack 结构在模型重命名后仍推导出相同宽度；
- deepstack 索引缺失、为 `None` 或为空时使用单份 `out_hidden_size`；
- 缺少 `out_hidden_size` 时使用 vLLM 通用接口；
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
- finish、abort、cancel 清理 `req_to_state`；
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
chunk_bytes / cache_namespace / pipeline / dp rank / save rank
```

日志禁止输出原始图片数据；identifier 只打印短前缀。

## 18. 分阶段实现

### 同步版本：正确性优先

- 配置期 layout resolver；
- 单 Store `Cache|Posix`；
- 固定行数 chunk；
- Scheduler 两张 map；
- `has_cache_item()` 基于 `lookup_on_prefix()` 的全 chunk 命中判断；
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

### Phase 3：Load failure、请求错误隔离与一致性增强

- 按第 11.6 节统一设计请求布局错误隔离与 load failure；此项尚未实施；
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

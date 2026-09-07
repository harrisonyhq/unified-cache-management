# UCM Encoder Cache Connector 同步版本 代码框架大纲

状态：Draft  
目标：vLLM V1 Engine，MRV2 为主、legacy MRV1 基础兼容  
Store：由 `ucm_connector_name` 选择的单个 Store；示例为 `Cache|Posix`  
同步版本 I/O：同步 `load/dump`  
最终目标：异步 dump、异步 load、Worker completion metadata、load failure 自动重算  
本期不包含：GDR、异步 Task、manifest Store

## 1. 同步版本 核心结论

```text
EC item
    identifier -> Tensor[N, D]

UCM representation
    identifier -> chunk_0 ... chunk_K-1
    每个 chunk 的逻辑 shape 固定为 [R, D]
    K = ceil(N / R)
```

- `D`、dtype 和 `R=rows_per_chunk` 在服务启动时确定。
- chunk ID 使用 UCM 现有 hash/block-id 方法生成，不新增 hash 实现。
- Scheduler 维护 `request -> identifiers` 和 `identifier -> chunk IDs`。
- `has_cache_item()` 每次调度 step 对 Store 做真实 lookup。
- `update_state_after_alloc()` 只把已经被 Scheduler 选中且外部命中的 item
  写入本 step metadata。
- Worker load 时分配完整 padded storage `[K,R,D]`，load 完后以
  `view(-1,D)[:N]` 放入 vLLM encoder cache。
- Worker dump 时 full chunk 直接引用原 tensor；最后一个不完整 chunk 使用
  局部 zero-padded tensor。
- 所有 Store I/O 在接口返回前 `wait()`，因此 同步版本 不需要 transfer ID、
  inflight table 或 Worker metadata。
- 同步版本 同步检测 load failure 并阻止无效 tensor 进入 encoder cache；自动重算在后续
  扩展 vLLM EC 调度协议后实现。

## 2. 文件与配置

```text
ucm/integration/vllm/ucm_ec_connector.py   # layout、metadata、Scheduler/Worker 接口
ucm/utils.py                            # Config.load_ec_config 与通用 YAML 读取
examples/ucm_ec_config.yaml              # EC 专用配置
```

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

以下代码是接口框架，不是完整可运行实现。

## 3. 公共数据结构

### 3.1 EncoderCacheLayout

```python
@dataclass(frozen=True, slots=True)
class EncoderCacheLayout:
    width: int
    dtype: torch.dtype
    rows_per_chunk: int
    chunk_bytes: int
```

启动期解析：

```python
def resolve_encoder_cache_layout(
    vllm_config: VllmConfig,
    encoder_config: dict[str, Any],
) -> EncoderCacheLayout:
    model_config = vllm_config.model_config
    width = encoder_config.get("encoder_cache_hidden_dim")
    if width is None:
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        output_width = getattr(vision_config, "out_hidden_size", None)
        if output_width is not None:
            deepstack_indexes = (
                getattr(vision_config, "deepstack_visual_indexes", None) or ()
            )
            width = output_width * (1 + len(deepstack_indexes))
        else:
            width = model_config.get_inputs_embeds_size()
    width = int(width)
    if width <= 0:
        raise EncoderCacheLayoutError(
            f"Encoder-cache width must be positive, got {width}."
        )

    dtype = model_config.dtype
    element_size = int(torch.empty((), dtype=dtype).element_size())

    rows_per_chunk = encoder_config["chunk_size"]
    if (
        not isinstance(rows_per_chunk, int)
        or isinstance(rows_per_chunk, bool)
        or rows_per_chunk <= 0
    ):
        raise EncoderCacheLayoutError(
            f"chunk_size must be a positive number of rows, got {rows_per_chunk!r}."
        )
    chunk_bytes = rows_per_chunk * width * element_size

    return EncoderCacheLayout(
        width=width,
        dtype=dtype,
        rows_per_chunk=rows_per_chunk,
        chunk_bytes=chunk_bytes,
    )

```

宽度、dtype、chunk 行数和字节数在同一个函数中解析。显式
`encoder_cache_hidden_dim` 优先；否则只要存在 `out_hidden_size` 就以它为基础输出宽度，
缺失、为 `None` 或为空的 `deepstack_visual_indexes` 按 0 层处理。没有
`out_hidden_size` 时使用 vLLM 的 `get_inputs_embeds_size()`。不按模型名称选择分支，也
不以通用输入宽度否定显式 EC 宽度。Worker 保存时仍校验实际 tensor。

### 3.2 Scheduler identifier state

```python
@dataclass(slots=True)
class IdentifierState:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]
    request_ref_count: int = 0
```

不重复保存以下可推导数据：

```text
num_chunks   = len(chunk_ids)
valid_bytes  = num_embeds * D * element_size
rows_per_chunk = connector 级常量
```

### 3.3 Scheduler -> Worker load metadata

```python
@dataclass(frozen=True, slots=True)
class ECLoadSpec:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]


@dataclass
class UCMECConnectorMetadata(ECConnectorMetadata):
    loads: dict[str, ECLoadSpec] = field(default_factory=dict)


class UCMEncoderCacheError(RuntimeError):
    """Runtime failure tied to a specific encoder-cache identifier."""

    def __init__(self, identifier: str, detail: str) -> None:
        super().__init__(f"{detail}: {identifier}")
        self.identifier = identifier
```

metadata 的 key 是 `request.mm_features[index].identifier`。

同步版本 不定义 `UCMECWorkerMetadata`：

```python
def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
    return None
```

这是阶段性实现，不是最终接口删除。Phase 2 会为 `ECLoadSpec` 增加 `transfer_id`，并新增
`UCMECWorkerMetadata`；完整框架见第 15 节。

## 4. Chunk ID 生成

```python
def make_chunk_ids(
    *,
    identifier: str,
    num_embeds: int,
    layout: EncoderCacheLayout,
    cache_namespace: bytes,
) -> tuple[bytes, ...]:
    num_chunks = ceil_div(num_embeds, layout.rows_per_chunk)

    return tuple(
        ucm_hash_block_id(
            cache_namespace,
            layout.rows_per_chunk,
            identifier,
            num_embeds,
            chunk_index,
        )
        for chunk_index in range(num_chunks)
    )
```

`ucm_hash_block_id()` 是对 UCM 当前 hash/block-id helper 的薄封装，负责转换成
UCM Store 已接受的固定长度 block ID，不在 EC connector 内引入另一种 hash 格式。

## 5. Connector 类框架

```python
class UCMECConnector(ECConnectorBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: ECConnectorRole,
    ) -> None:
        super().__init__(vllm_config, role)
        mm_config = vllm_config.model_config.multimodal_config
        if mm_config is not None and mm_config.is_multimodal_pruning_enabled():
            raise EncoderCacheLayoutError("UCM EC does not support multimodal pruning.")

        ec_config = Config.load_ec_config(vllm_config.ec_transfer_config)
        encoder_config = ec_config["encoder_cache_config"]
        self._block_hasher = RequestHasher(vllm_config, 0)
        self.layout = resolve_encoder_cache_layout(
            vllm_config,
            encoder_config,
            self._block_hasher,
        )
        self.cache_namespace = resolve_cache_namespace(
            vllm_config,
            encoder_config,
            self._block_hasher,
        )

        # 复用现有 UCM connector 的 rank/device 初始化。Scheduler 不分配
        # tensor，因此保留 -1/None；Worker 直接调用 UCM 共用方法。
        self.local_rank = (
            -1
            if role == ECConnectorRole.SCHEDULER
            else get_world_group().local_rank
        )
        self.device_id = self.local_rank
        self.device: torch.device | None = (
            None
            if role == ECConnectorRole.SCHEDULER
            else get_ucm_worker_torch_device(self.local_rank)
        )

        # 每个进程各自创建 handle，但连接同一个配置指定的逻辑 Store。
        self.store = create_ucm_ec_store(
            vllm_config=vllm_config,
            role=role,
            layout=self.layout,
            ec_config=ec_config,
            device_id=self.device_id,
        )

        # Scheduler-only persistent state.
        self.req_to_identifiers: dict[str, tuple[str, ...]] = {}
        self.identifier_to_blocks: dict[str, IdentifierState] = {}

        # Scheduler-only, current schedule step state.
        self.step_verified_hits: set[str] = set()
        self.pending_loads: dict[str, ECLoadSpec] = {}

        # Worker-only state.
        self.is_save_rank = (
            role == ECConnectorRole.WORKER
            and resolve_is_save_rank(vllm_config)
        )
```

`get_ucm_worker_torch_device()` 不在 EC connector 中重新实现。落地时把当前
`UCMDirectConnector.__init__` 中的 CUDA-alike/NPU 分支提取为
`ucm.integration.vllm.device` 的公共 helper，并让 KV connector 与 EC connector 共同
调用。不要让 `UCMECConnector` 继承整个 `UCMDirectConnector`：后者继承
`KVConnectorBase_V1`，其初始化、metadata 和 KV layout 生命周期都不适用于 EC。

这里也不能复用当前 `ucm.integration.vllm.device.create_device()`：该方法返回的是 UCM
用于 CPU affinity/NUMA 管理的 `Device` 对象，不是分配 encoder tensor 所需的
`torch.device`。

`resolve_is_save_rank()` 的首版规则：

```text
持有 multimodal encoder 的首个 PP stage
&& tp_rank == 0
&& pcp_rank == 0
```

## 6. Scheduler 侧接口

### 6.1 `ensure_cache_available()`

调用时机：waiting request 被 Scheduler 尝试接纳为 running 时调用。在 chunked
prefill 或 preemption 后，同一个 request 可能再次经过这里，因此必须幂等。

```python
def ensure_cache_available(
    self,
    request: Request,
    num_computed_tokens: int,
) -> bool:
    req_id = request.request_id

    if req_id in self.req_to_identifiers:
        return True

    # 第一遍：遍历媒体 item，去重、校验并构造待提交 state。
    request_states: dict[str, IdentifierState] = {}

    for index, feature in enumerate(request.mm_features):
        identifier = feature.identifier
        num_embeds = request.get_num_encoder_embeds(index)

        state = request_states.get(identifier)
        if state is not None:
            # 同一 request 内，相同 identifier 只能对应相同 N。
            if state.num_embeds != num_embeds:
                raise EncoderCacheLayoutError(...)
            continue

        state = self.identifier_to_blocks.get(identifier)
        if state is not None:
            # 其他活跃 request 已经注册该 identifier。
            if state.num_embeds != num_embeds:
                raise EncoderCacheLayoutError(...)
        else:
            state = IdentifierState(
                num_embeds=num_embeds,
                chunk_ids=make_chunk_ids(
                    identifier=identifier,
                    num_embeds=num_embeds,
                    layout=self.layout,
                    cache_namespace=self.cache_namespace,
                ),
            )

        request_states[identifier] = state

    # 第二遍只遍历去重后的 U 个 identifier，提交并增加 request ref count。
    identifiers = tuple(request_states)
    self.req_to_identifiers[req_id] = identifiers

    for identifier, candidate in request_states.items():
        state = self.identifier_to_blocks.setdefault(
            identifier,
            candidate,
        )
        state.request_ref_count += 1

    return True
```

`num_computed_tokens` 是 vLLM 接口参数；同步版本 不需要使用。

复杂度：第一次 `O(M)`，第二次 `O(U)`，其中 `U <= M`。

### 6.2 `has_cache_item()`

调用时机：Scheduler 扫描当前 token window 中需要 encoder output 的 item 时调用。

```python
def has_cache_item(self, identifier: str) -> bool:
    if not self.is_consumer:
        return False

    state = self.identifier_to_blocks.get(identifier)
    if state is None:
        return False

    found = self.store.lookup(list(state.chunk_ids))
    hit = len(found) == len(state.chunk_ids) and all(found)

    if hit:
        self.step_verified_hits.add(identifier)

    return hit
```

同步版本 只保存本 step 的 positive hit：

```text
hit  -> 加入 step_verified_hits
miss -> 不保存
```

`step_verified_hits` 会在 `build_connector_meta()` 中清空，因此不会把上一次调度
命中带到下一次调度。

### 6.3 `update_state_after_alloc()`

调用时机：Scheduler 已经决定本 step 处理该 item，并且
`encoder_cache_manager.allocate(request, index)` 成功之后。

```python
def update_state_after_alloc(
    self,
    request: Request,
    index: int,
) -> None:
    identifier = request.mm_features[index].identifier

    # compute miss 路径同样会调用该接口，但不需要外部 load。
    if identifier not in self.step_verified_hits:
        return

    # 同一个 step 内，相同 identifier 只发送一份 load metadata。
    if identifier in self.pending_loads:
        return

    state = self.identifier_to_blocks.get(identifier)
    if state is None:
        # 不能静默 return：该 item 已走 external-load 路径，没有 compute 兜底。
        raise UCMEncoderCacheError(
            identifier,
            "Missing scheduler state for verified encoder cache item",
        )

    self.pending_loads[identifier] = ECLoadSpec(
        num_embeds=state.num_embeds,
        chunk_ids=state.chunk_ids,
    )
```

这个接口不做 lookup，也不执行 I/O。它只是把“已经 allocate 且确定外部命中”的
item 转换成 Scheduler -> Worker metadata。

### 6.4 `build_connector_meta()`

调用时机：Scheduler 已完成本 step 所有 request 的调度、准备构造
`SchedulerOutput` 时调用。

```python
def build_connector_meta(
    self,
    scheduler_output: SchedulerOutput,
) -> UCMECConnectorMetadata:
    metadata = UCMECConnectorMetadata(
        loads=self.pending_loads,
    )

    # metadata 接管旧 dict；connector 为下一个 step 创建新 dict。
    self.pending_loads = {}
    self.step_verified_hits.clear()

    # 即使 loads 为空也返回 metadata，而不是 None。
    return metadata
```

返回空 metadata 的原因是 producer Worker 仍需进入 EC wrapper，从而在 encoder
forward 后收到 `save_caches()` 调用。

### 6.5 `request_finished()`

调用时机：request 最终完成、abort 或 cancel，并且 vLLM 即将释放其 encoder cache
引用时调用。Preemption 不属于 request finish。

```python
def request_finished(
    self,
    request: Request,
) -> tuple[bool, dict[str, Any] | None]:
    identifiers = self.req_to_identifiers.pop(
        request.request_id,
        None,
    )
    if identifiers is None:
        return False, None

    for identifier in identifiers:
        state = self.identifier_to_blocks.get(identifier)
        if state is None:
            continue

        state.request_ref_count -= 1
        if state.request_ref_count == 0:
            del self.identifier_to_blocks[identifier]

    # 同步版本 同步 I/O，不延迟 request 释放。
    return False, None
```

这里只清 Scheduler 内存状态，不删除 UCM chunk。

### 6.6 Scheduler 侧 no-op 接口

```python
def update_connector_output(
    self,
    connector_output: ECConnectorOutput,
) -> None:
    # 同步版本 没有 Worker metadata。
    return


def has_pending_push_work(self) -> bool:
    # 同步版本 没有异步任务需要 scheduler heartbeat。
    return False
```

## 7. Worker 侧接口

### 7.1 metadata 绑定

`bind_connector_metadata()` 和 `clear_connector_metadata()` 直接使用
`ECConnectorBase` 的实现：

```text
SchedulerOutput.ec_connector_metadata
    -> Worker bind_connector_metadata()
    -> start_load_caches()/save_caches() 读取
    -> clear_connector_metadata()
```

### 7.2 `start_load_caches()`

调用时机：Worker model forward 之前、读取 multimodal encoder output 之前。

```python
def start_load_caches(
    self,
    encoder_cache: dict[str, torch.Tensor],
    **kwargs: Any,
) -> None:
    metadata = self._get_connector_metadata()
    assert isinstance(metadata, UCMECConnectorMetadata)
    assert self.device is not None

    rows = self.layout.rows_per_chunk
    width = self.layout.width

    for identifier, spec in metadata.loads.items():
        # 同步 同步版本 不需要 completion ack；本地已存在即可跳过。
        if identifier in encoder_cache:
            continue

        num_chunks = len(spec.chunk_ids)

        # 分配完整 padded storage。最后一个 chunk 即使只有部分有效行，
        # UCM 仍然可以写入完整 [R,D]。
        storage = torch.empty(
            (num_chunks, rows, width),
            dtype=self.layout.dtype,
            device=self.device,
        )

        dst_chunks = [
            [storage[chunk_index]]
            for chunk_index in range(num_chunks)
        ]

        try:
            task = self.store.load(
                list(spec.chunk_ids),
                [0] * num_chunks,
                dst_chunks,
            )
            self.store.wait(task)
        except Exception as exc:
            # storage 尚未插入 encoder_cache，不会被模型消费。
            encoder_cache.pop(identifier, None)
            raise UCMEncoderCacheError(
                identifier, "Failed to load encoder cache item"
            ) from exc

        # 截断是 view；底层 padded storage 会随 output 一起保留。
        output = storage.view(-1, width)[: spec.num_embeds]
        encoder_cache[identifier] = output
```

重要顺序：只有 `store.wait()` 成功后才能把 output 放进 `encoder_cache`。同步版本 的
load failure 会同步终止本次执行；由于 Scheduler 已经选择 external-load 路径，没有安排
encoder compute，connector 不能在 Worker 内直接静默重算。

在当前 vLLM 调用链中，该异常会从 `start_load_caches()` 穿过 model runner 的
`execute_model()`，最终由 EngineCore busy loop 视为 fatal error。多进程服务中 HTTP/API
进程不一定立即退出，但 EngineCore 会被标记为 dead，当前及后续请求失败，通常需要重启
服务。因此 同步版本 的语义是“服务级 fail-closed”，不是请求级失败或自动重算。

这里的 `output` 虽然逻辑 shape 是 `[N,D]`，但它仍引用完整的
`[ceil(N/R),R,D]` storage。令 `L` 为 Worker encoder cache 中由外部 load 得到的活跃
item 集合，则：

```text
Scheduler_accounted_bytes(L) = sum(N_i * D * element_size)
Worker_allocated_bytes(L) = sum(ceil(N_i / R) * R * D * element_size)
0 <= Worker_allocated_bytes - Scheduler_accounted_bytes
  < len(L) * chunk_bytes
```

本地 encoder compute 产生的 `[N,D]` tensor 不承担这笔 padding。同步版本 接受该显存
记账偏差；容量规划需要为最坏情况下每个外部 load 活跃 item 预留不到一个 chunk。

### 7.3 `save_caches()`

调用时机：

- MRV2：encoder forward 后，wrapper 比较前后 encoder cache keys，对新增 entry 调用；
- legacy MRV1：每个 encoder output 写入本地 cache 后立即调用。

connector 内部必须自行进行 producer 和唯一 writer 门控：

```python
def save_caches(
    self,
    encoder_cache: dict[str, torch.Tensor],
    mm_hash: str,
    **kwargs: Any,
) -> None:
    if not self.is_producer:
        return
    if not self.is_save_rank:
        return

    tensor = encoder_cache[mm_hash]
    self._validate_encoder_tensor(tensor)

    num_embeds = tensor.shape[0]
    rows = self.layout.rows_per_chunk
    width = self.layout.width

    chunk_ids = make_chunk_ids(
        identifier=mm_hash,
        num_embeds=num_embeds,
        layout=self.layout,
        cache_namespace=self.cache_namespace,
    )

    num_full_chunks, tail_rows = divmod(num_embeds, rows)
    src_chunks: list[list[torch.Tensor]] = [
        [tensor[i * rows : (i + 1) * rows]]
        for i in range(num_full_chunks)
    ]

    # 只有 dump 的最后一个不完整 chunk 需要局部 padded tensor。
    tail: torch.Tensor | None = None
    if tail_rows:
        tail = torch.zeros(
            (rows, width),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tail[:tail_rows].copy_(tensor[num_full_chunks * rows :])
        src_chunks.append([tail])

    try:
        task = self.store.dump(
            list(chunk_ids),
            [0] * len(chunk_ids),
            src_chunks,
        )
        self.store.wait(task)
    except Exception:
        # Dump 是旁路缓存优化，失败不影响当前 encoder 计算结果。
        logger.exception("Failed to dump EC item %s", mm_hash)
        return

    # 同步 wait 返回后，局部 tail 和 tensor view 可以安全释放。
```

建议的最小校验：

```python
def _validate_encoder_tensor(self, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise EncoderCacheLayoutError(...)
    if tensor.shape[1] != self.layout.width:
        raise EncoderCacheLayoutError(...)
    if tensor.dtype != self.layout.dtype:
        raise EncoderCacheLayoutError(...)
    if not tensor.is_contiguous():
        raise EncoderCacheLayoutError(...)
```

### 7.4 Worker 侧 no-op 接口

```python
def register_caches(
    self,
    ec_caches: dict[str, torch.Tensor],
) -> None:
    return


def get_finished(
    self,
    finished_req_ids: set[str],
) -> tuple[set[str] | None, set[str] | None]:
    return None, None


def build_connector_worker_meta(
    self,
) -> ECConnectorWorkerMetadata | None:
    return None
```

### 7.5 `shutdown()`

```python
def shutdown(self) -> None:
    self.pending_loads.clear()
    self.step_verified_hits.clear()
    self.req_to_identifiers.clear()
    self.identifier_to_blocks.clear()

    if self.store is not None:
        self.store.close()
```

同步版本 所有 I/O 已同步完成，shutdown 不需要处理 inflight Task。

## 8. Scheduler 调用时序

### 8.1 request 从 waiting 进入调度

vLLM Scheduler 调用链：

```text
schedule()
  |
  |-- 尝试接纳 waiting request
  |     `-- ec_connector.ensure_cache_available(request, num_computed_tokens)
  |
  |-- _try_schedule_encoder_inputs()
  |     |
  |     `-- ec_connector.has_cache_item(identifier)
  |            |
  |            |-- hit  -> external_load_encoder_input.append(index)
  |            `-- miss -> encoder_inputs_to_schedule.append(index)
  |
  |-- 对真正选中的 item 分配本地 encoder cache 容量
  |     |
  |     |-- encoder_cache_manager.allocate(request, index)
  |     `-- ec_connector.update_state_after_alloc(request, index)
  |
  `-- 构造 SchedulerOutput
        `-- scheduler_output.ec_connector_metadata =
                ec_connector.build_connector_meta(scheduler_output)
```

源码调用点：

```text
ensure_cache_available   scheduler.py:917
has_cache_item           scheduler.py:1741
allocate + update        scheduler.py:753-762、1196-1206
build_connector_meta     scheduler.py:1367-1372
request_finished         scheduler.py:2493-2497
```

### 8.2 `update_state_after_alloc()` 的上下文

```text
has_cache_item() 只是判断 Store 是否命中
        |
        v
Scheduler 继续进行 budget/capacity 判断
        |
        v
allocate() 成功，说明该 item 确实进入本 step
        |
        v
update_state_after_alloc()
        |
        `-- 若 identifier 在 step_verified_hits 中：
              pending_loads[identifier] = ECLoadSpec(...)
```

因此不能在 `has_cache_item()` 内直接创建 load metadata；lookup 命中不等于该 item
最终一定被本 step 调度。

## 9. Worker 调用时序

### 9.1 MRV2 主路径

```text
ActiveECConnector.maybe_get_output(scheduler_output)
  |
  |-- bind_connector_metadata(metadata)
  |
  |-- consumer:
  |     `-- start_load_caches(encoder_cache)
  |           `-- store.load() + wait()
  |
  |-- cached_hashes = set(encoder_cache)
  |
  |-- multimodal encoder/model forward
  |
  |-- producer:
  |     `-- 对 encoder_cache.keys() - cached_hashes
  |           `-- save_caches(encoder_cache, mm_hash)
  |                 `-- store.dump() + wait()
  |
  |-- get_finished() -> (None, None)
  |-- build_connector_worker_meta() -> None
  `-- clear_connector_metadata()
```

外部 load 发生在记录 `cached_hashes` 之前，所以 load 进来的 entry 不会被当成新计算
结果再次 dump。

### 9.2 Legacy MRV1 路径

```text
ECConnectorModelRunnerMixin
  |
  |-- bind_connector_metadata(metadata)
  |-- consumer -> start_load_caches()
  |-- encoder forward
  |-- encoder_cache[mm_hash] = output
  |-- maybe_save_ec_to_connector()
  |     `-- save_caches(encoder_cache, mm_hash)
  |-- get_finished() -> (None, None)
  `-- clear_connector_metadata()
```

legacy wrapper 不做 producer 门控，因此 `save_caches()` 内部的 `is_producer` 和
`is_save_rank` 检查不能省略。legacy wrapper 当前不回传 Worker metadata，但 同步版本
本身不使用 Worker metadata，因此不影响同步路径。

## 10. Metadata 数据流向

```text
Scheduler process

identifier_to_blocks
    identifier -> IdentifierState(N, chunk_ids, ref_count)
                |
                | has_cache_item() 命中
                v
step_verified_hits
                |
                | update_state_after_alloc()
                v
pending_loads
    identifier -> ECLoadSpec(N, chunk_ids)
                |
                | build_connector_meta()
                v
UCMECConnectorMetadata(loads=...)
                |
                | SchedulerOutput 序列化/分发
                v
Worker process(es)

bind_connector_metadata()
                |
                v
start_load_caches(encoder_cache)
                |
                | store.load(chunk_ids)
                v
encoder_cache[identifier] = Tensor[N,D]
```

反向保存路径不需要 Scheduler metadata：

```text
Worker encoder forward
    -> encoder_cache[identifier] = Tensor[N,D]
    -> wrapper 调用 save_caches()
    -> Worker 根据实际 N 重新生成相同 chunk_ids
    -> store.dump(chunk_ids)
```

## 11. 状态生命周期

| 状态 | 所在进程 | 生命周期 | 清理点 |
|---|---|---|---|
| `req_to_identifiers` | Scheduler | request 生命周期 | `request_finished()` |
| `identifier_to_blocks` | Scheduler | 至少一个活跃 request 引用期间 | ref count 变为 0 |
| `step_verified_hits` | Scheduler | 单个 schedule step | `build_connector_meta()` |
| `pending_loads` | Scheduler | 单个 schedule step | 移交给 metadata 后换新 dict |
| `UCMECConnectorMetadata` | Scheduler -> Worker | 单个 model execution step | Worker clear metadata |
| padded load storage | Worker | vLLM 本地 encoder cache 生命周期 | vLLM 释放对应 mm hash |
| dump tail tensor | Worker | 单次同步 `save_caches()` | `wait()` 返回后 |

## 12. 两条端到端路径

### 12.1 外部命中

```text
ensure request state
    -> Store lookup 所有 chunk 均命中
    -> step_verified_hits.add(identifier)
    -> Scheduler allocate
    -> update_state_after_alloc
    -> metadata.loads
    -> Worker 分配 padded storage
    -> Store load + wait
    -> encoder_cache[identifier] = storage.view(-1,D)[:N]
    -> 模型消费 encoder output
```

### 12.2 外部 miss 后保存

```text
ensure request state
    -> Store lookup miss
    -> Scheduler allocate 并安排 encoder compute
    -> update_state_after_alloc no-op
    -> Worker 执行 encoder
    -> encoder_cache[identifier] = Tensor[N,D]
    -> wrapper 调用 save_caches
    -> full chunks 使用原 tensor view
    -> tail chunk zero padding
    -> Store dump + wait
```

## 13. 同步版本 接口总表

| 接口 | 角色 | 同步版本 行为 |
|---|---|---|
| `__init__` | 两侧 | 解析 layout，创建同一逻辑 Store 的进程内 handle |
| `ensure_cache_available` | Scheduler | 两遍建立 request/identifier state，幂等 |
| `has_cache_item` | Scheduler | lookup 全部 chunk，命中写入 step hit set |
| `update_state_after_alloc` | Scheduler | allocate 后将 hit item 加入 pending loads |
| `build_connector_meta` | Scheduler | 生成 load metadata，清空 step 状态 |
| `request_finished` | Scheduler | 清 request map，递减 identifier ref count |
| `update_connector_output` | Scheduler | no-op |
| `has_pending_push_work` | Scheduler | `False` |
| `bind_connector_metadata` | Worker | 使用基类实现 |
| `start_load_caches` | Worker | padded storage，同步 load，完成后插入 cache |
| `save_caches` | Worker | producer/rank 门控，同步 dump，尾块 padding |
| `register_caches` | Worker | no-op |
| `get_finished` | Worker | `(None, None)` |
| `build_connector_worker_meta` | Worker | `None` |
| `clear_connector_metadata` | Worker | 使用基类实现 |
| `shutdown` | 两侧 | 清本地状态并关闭 Store handle |

## 14. 同步版本 最小测试集合

```text
1. 相同 identifier/N/layout 生成稳定且符合 UCM 格式的 chunk IDs。
2. ensure 重复调用不重复增加 ref count。
3. 同 request 重复 identifier 只增加一次 ref count。
4. 相同 identifier 对应不同 N 时拒绝建立状态。
5. 全 chunk 命中才返回 true，部分命中返回 false。
6. build_connector_meta 后 step_verified_hits 被清空。
7. update_state_after_alloc 只为 hit item 生成 load metadata。
8. N 可整除 R 时，load/dump 无 tail。
9. N 不可整除 R 时，load padded view 的前 N 行正确。
10. dump 尾块 padding 为零且不越界读取原 tensor。
11. TP/PCP 多 rank 下只有唯一 save rank 执行 dump。
12. MRV2 diff-save 和 legacy MRV1 eager-save 均通过 connector 内部门控。
13. finish/abort/cancel 正确减少 ref count，preemption 不删除 request state。
14. load/wait 失败时不向 encoder cache 插入任何 output，并抛出明确错误。
15. dump/wait 失败只记录错误，不影响当前已经完成 encoder compute 的请求。
```

## 15. 异步与 Load Failure 演进框架

本节是后续实现规划，不属于当前同步实现。演进顺序固定为：

```text
同步版本   同步 load/dump + 同步 load failure 检测
Phase 2A  异步 dump
Phase 2B  异步 load + stream/event 依赖
Phase 3   load failure 后自动 encoder recompute
```

### 15.1 Phase 2 公共 metadata 扩展

Scheduler 为每次 load dispatch 分配单调 `transfer_id`：

```python
@dataclass(frozen=True, slots=True)
class ECLoadSpec:
    transfer_id: int
    num_embeds: int
    chunk_ids: tuple[bytes, ...]
```

Worker 使用以下状态持有尚未完成的 UCM Task 和地址生命周期：

```python
@dataclass(slots=True)
class InflightECTransfer:
    task: UCMTask
    identifier: str
    transfer_id: int | None
    direction: Literal["load", "dump"]
    held_tensors: tuple[torch.Tensor, ...]
    output: torch.Tensor | None = None
```

Worker -> Scheduler metadata：

```python
@dataclass
class UCMECWorkerMetadata(ECConnectorWorkerMetadata):
    # replicated load：同一 transfer_id 会由参与的 TP/PCP rank 分别报告。
    completed_load_ids: list[int] = field(default_factory=list)
    failed_loads: list[tuple[int, str, str]] = field(default_factory=list)

    # dump 只有唯一 save rank 报告。
    completed_dump_ids: list[str] = field(default_factory=list)
    failed_dumps: list[tuple[str, str]] = field(default_factory=list)

    def aggregate(
        self,
        other: ECConnectorWorkerMetadata,
    ) -> ECConnectorWorkerMetadata:
        assert isinstance(other, UCMECWorkerMetadata)
        self.completed_load_ids.extend(other.completed_load_ids)
        self.failed_loads.extend(other.failed_loads)
        self.completed_dump_ids.extend(other.completed_dump_ids)
        self.failed_dumps.extend(other.failed_dumps)
        return self
```

### 15.2 Phase 2A：异步 dump

`save_caches()` 不再立即 `wait()`：

```python
task = self.store.dump(...)
self.inflight_dumps[identifier] = InflightECTransfer(
    task=task,
    identifier=identifier,
    transfer_id=None,
    direction="dump",
    held_tensors=(tensor, *tail_tensors),
)
```

必须由 inflight entry 持有原 encoder tensor、full chunk views 和 tail tensor，直到 Task
完成。`build_connector_worker_meta()` 轮询：

```python
def build_connector_worker_meta(self) -> UCMECWorkerMetadata | None:
    meta = UCMECWorkerMetadata()

    for identifier, transfer in list(self.inflight_dumps.items()):
        if not self.store.check(transfer.task):
            continue

        try:
            self.store.wait(transfer.task)
        except Exception as exc:
            meta.failed_dumps.append((identifier, str(exc)))
        else:
            meta.completed_dump_ids.append(identifier)

        del self.inflight_dumps[identifier]

    return meta if not is_empty(meta) else None
```

dump failure 只用于指标和诊断，不使用户请求失败，也不生成长期命中状态；后续仍由
`has_cache_item()` lookup 判断是否成功写入 Store。

### 15.3 Phase 2B：异步 load

Worker 为 metadata 中的每个 load 创建 padded storage，并提交 UCM Task：

```python
task = self.store.load(...)
self.inflight_loads[spec.transfer_id] = InflightECTransfer(
    task=task,
    identifier=identifier,
    transfer_id=spec.transfer_id,
    direction="load",
    held_tensors=(storage,),
    output=storage.view(-1, width)[: spec.num_embeds],
)
```

异步 load 的正确性条件不是“去掉 `wait()`”，而是：

```text
UCM load stream/event
    -> model compute stream wait
    -> 模型才允许读取 encoder output
```

如果 UCM Task 无法提供可供 compute stream 等待的 event/stream dependency，
`start_load_caches()` 仍必须同步 `wait()`，不能让模型读取尚未完成的 storage。

`build_connector_worker_meta()` 轮询 load Task，并通过 `completed_load_ids` 或
`failed_loads` 报告结果。Scheduler 的 `update_connector_output()` 按 `transfer_id` 汇总
参与 TP/PCP rank 的结果；任一 rank 失败，该 transfer 即失败。

### 15.4 Scheduler 异步接口变化

```python
class UCMECConnector(ECConnectorBase):
    def update_connector_output(
        self,
        connector_output: ECConnectorOutput,
    ) -> None:
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, UCMECWorkerMetadata):
            return

        update_load_ack_counts(meta.completed_load_ids)
        record_load_failures(meta.failed_loads)
        record_dump_results(
            meta.completed_dump_ids,
            meta.failed_dumps,
        )

    def has_pending_push_work(self) -> bool:
        return bool(self.scheduler_visible_inflight)
```

`has_pending_push_work()` 用于没有其他 runnable request 时继续触发 Worker no-forward，
使 `build_connector_worker_meta()` 仍有机会轮询并返回完成事件。

### 15.5 Phase 3：Load failure 自动重算

仅仅在 `UCMECWorkerMetadata` 中增加 `failed_loads`，只能做到检测和上报，不能直接完成
重算。原因是 Scheduler 在 `has_cache_item()` 命中后已经：

```text
把 item 放入 external-load 路径
没有扣 encoder compute budget
没有把 item 放入 encoder_inputs_to_schedule
```

完整恢复需要扩展 vLLM EC Scheduler/Model Runner 协议：

```text
Worker 在模型消费前确认 load failure
    -> 返回 failed transfer_id / identifier
    -> Scheduler 释放失败的 encoder cache allocation
    -> 清除本次 external-hit 状态
    -> 下一 step 强制把该 identifier 视为 external miss
    -> 重新进入 encoder_inputs_to_schedule
    -> Worker 正常执行 multimodal encoder
```

同步版本 的 `UCMEncoderCacheError` 保证错误 tensor 不被消费；Phase 3 才把“请求失败”
升级成“透明 encoder recompute”。

### 15.6 异步实现测试增量

```text
1. Task 完成前 inflight entry 持有所有源/目标 tensor。
2. 异步 dump completion/failure 由唯一 save rank 上报。
3. 多 TP/PCP rank 聚合时保留重复 completed_load_ids。
4. 任一 rank load failure 使 transfer 失败。
5. load stream 未完成时 compute stream 不得读取 output。
6. stale/重复 transfer report 不得影响后续同 identifier 的新 transfer。
7. has_pending_push_work 在 inflight 清空前保持 engine polling。
8. load failure 自动重算时，失败 tensor 不进入模型，下一 step 执行 encoder compute。
```

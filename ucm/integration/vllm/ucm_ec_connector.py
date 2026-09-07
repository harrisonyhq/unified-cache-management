"""UCM-backed vLLM encoder-cache connector.

The connector uses synchronous Store I/O. An encoder-cache item is a
variable-length ``[N, D]`` tensor, while UCM stores fixed-size blocks, so each
item is represented by one or more fixed-row chunks. Scheduler state carries
only the chunk IDs required by a selected external hit; workers reconstruct
the tensor from padded chunk storage or dump a newly computed tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_world_group,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

from ucm.integration.vllm.device import get_ucm_worker_torch_device
from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.logger import init_logger
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class EncoderCacheLayoutError(ValueError):
    """Raised when scheduler and worker cannot share one fixed EC layout."""


class UCMEncoderCacheError(RuntimeError):
    """Runtime failure tied to a specific encoder-cache identifier."""

    def __init__(self, identifier: str, detail: str) -> None:
        super().__init__(f"{detail}: {identifier}")
        self.identifier = identifier


@dataclass(frozen=True, slots=True)
class EncoderCacheLayout:
    """EncoderCache tensor shape: [N,D]"""

    # Embedding width per row in encoder cache, D.
    width: int
    dtype: torch.dtype
    rows_per_chunk: int
    chunk_bytes: int
    layout_id: bytes


@dataclass(slots=True)
class IdentifierState:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]
    request_ref_count: int = 0


@dataclass(frozen=True, slots=True)
class ECLoadSpec:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]


@dataclass
class UCMECConnectorMetadata(ECConnectorMetadata):
    loads: dict[str, ECLoadSpec] = field(default_factory=dict)


def resolve_encoder_cache_layout(
    vllm_config: VllmConfig,
    encoder_config: dict[str, Any],
    hasher: RequestHasher | None = None,
) -> EncoderCacheLayout:
    model_config = vllm_config.model_config
    width = encoder_config.get("encoder_cache_hidden_dim")
    if width is None:
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        output_width = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if output_width is not None and deepstack_indexes is not None:
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

    block_hasher = hasher or RequestHasher(vllm_config, 0)
    layout_id = block_hasher(
        ("ucm-ec-layout-v1", width, str(dtype), rows_per_chunk)
    )
    return EncoderCacheLayout(
        width=width,
        dtype=dtype,
        rows_per_chunk=rows_per_chunk,
        chunk_bytes=chunk_bytes,
        layout_id=layout_id,
    )


def resolve_cache_namespace(
    vllm_config: VllmConfig,
    encoder_config: dict[str, Any],
    hasher: RequestHasher,
) -> bytes:
    explicit = encoder_config.get("cache_namespace")
    if explicit is not None:
        namespace_source: Any = ("explicit", str(explicit))
    else:
        model_config = vllm_config.model_config
        mm_config = model_config.multimodal_config
        mm_hash = mm_config.compute_hash() if mm_config is not None else None
        hf_config = model_config.hf_config
        namespace_source = (
            "inferred",
            (
                model_config.model,
                model_config.revision,
                model_config.code_revision,
                model_config.tokenizer_revision,
                getattr(hf_config, "_commit_hash", None),
                tuple(getattr(hf_config, "architectures", None) or ()),
                mm_hash,
            ),
        )
    return hasher(("ucm-ec-namespace-v1", namespace_source))


def make_chunk_ids(
    *,
    identifier: str,
    num_embeds: int,
    layout: EncoderCacheLayout,
    cache_namespace: bytes,
    hasher: RequestHasher,
) -> tuple[bytes, ...]:
    if not identifier:
        raise EncoderCacheLayoutError("Encoder-cache identifier must not be empty.")
    if num_embeds <= 0:
        raise EncoderCacheLayoutError(
            f"Encoder-cache item {identifier!r} must have at least one row, "
            f"got {num_embeds}."
        )
    num_chunks = (num_embeds + layout.rows_per_chunk - 1) // layout.rows_per_chunk
    return tuple(
        hasher(
            (
                "ucm-ec-v1",
                cache_namespace,
                layout.layout_id,
                identifier,
                num_embeds,
                chunk_index,
            )
        )
        for chunk_index in range(num_chunks)
    )


def create_ucm_ec_store(
    *,
    vllm_config: VllmConfig,
    role: ECConnectorRole,
    layout: EncoderCacheLayout,
    ec_config: dict[str, Any],
    device_id: int,
) -> UcmKVStoreBaseV1:
    name = ec_config["ucm_connector_name"]
    store_config = dict(ec_config["ucm_connector_config"])
    encoder_config = ec_config["encoder_cache_config"]
    if "use_gdr" in store_config:
        logger.warning("UCM EC ignores configured use_gdr; GDR is disabled.")

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    instance_id = vllm_config.instance_id or vllm_config.ec_transfer_config.engine_id
    unique_id = encoder_config.get("store_unique_id")
    if unique_id is None:
        unique_id = f"{instance_id}.ec.dp{dp_rank}"

    store_config.update(
        {
            "unique_id": unique_id,
            "device_id": device_id,
            "share_buffer_enable": True,
            "local_rank_size": parallel_config.tensor_parallel_size,
            "use_gdr": False,
            "tensor_size_list": [layout.chunk_bytes],
            "shard_size": layout.chunk_bytes,
            "block_size": layout.chunk_bytes,
            "posix_gc_enable": (
                role == ECConnectorRole.SCHEDULER and dp_rank == 0
            ),
        }
    )
    logger.info(
        "Creating UCM EC store %s: unique_id=%s, device_id=%s, chunk_bytes=%s",
        name,
        unique_id,
        device_id,
        layout.chunk_bytes,
    )
    return UcmConnectorFactoryV1.create_connector(name, store_config)


class UCMECConnector(ECConnectorBase):
    """Synchronous, fixed-chunk UCM encoder-cache connector."""

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
            vllm_config, encoder_config, self._block_hasher
        )
        self.cache_namespace = resolve_cache_namespace(
            vllm_config, encoder_config, self._block_hasher
        )

        self.local_rank = (
            -1
            if role == ECConnectorRole.SCHEDULER
            else int(get_world_group().local_rank)
        )
        self.device_id = self.local_rank
        self.device = (
            None
            if role == ECConnectorRole.SCHEDULER
            else get_ucm_worker_torch_device(self.local_rank)
        )
        self.store: UcmKVStoreBaseV1 | None = create_ucm_ec_store(
            vllm_config=vllm_config,
            role=role,
            layout=self.layout,
            ec_config=ec_config,
            device_id=self.device_id,
        )

        self.req_to_identifiers: dict[str, tuple[str, ...]] = {}
        self.identifier_to_blocks: dict[str, IdentifierState] = {}
        self.step_verified_hits: set[str] = set()
        self.pending_loads: dict[str, ECLoadSpec] = {}

        self.is_save_rank = False
        if role == ECConnectorRole.WORKER:
            self.is_save_rank = (
                get_pp_group().is_first_rank
                and get_tensor_model_parallel_rank() == 0
                and get_pcp_group().rank_in_group == 0
            )

        logger.info(
            "Initialized UCM EC connector: role=%s, producer=%s, consumer=%s, "
            "width=%s, dtype=%s, rows_per_chunk=%s, chunk_bytes=%s, "
            "layout_id=%s, cache_namespace=%s, save_rank=%s",
            role.name,
            self.is_producer,
            self.is_consumer,
            self.layout.width,
            self.layout.dtype,
            self.layout.rows_per_chunk,
            self.layout.chunk_bytes,
            self.layout.layout_id.hex(),
            self.cache_namespace.hex(),
            self.is_save_rank,
        )

    def _make_chunk_ids(
        self, identifier: str, num_embeds: int
    ) -> tuple[bytes, ...]:
        return make_chunk_ids(
            identifier=identifier,
            num_embeds=num_embeds,
            layout=self.layout,
            cache_namespace=self.cache_namespace,
            hasher=self._block_hasher,
        )

    def ensure_cache_available(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> bool:
        del num_computed_tokens
        req_id = request.request_id
        if req_id in self.req_to_identifiers:
            return True

        request_states: dict[str, IdentifierState] = {}
        for index, feature in enumerate(request.mm_features):
            identifier = feature.identifier
            num_embeds = int(request.get_num_encoder_embeds(index))

            state = request_states.get(identifier)
            if state is not None:
                if state.num_embeds != num_embeds:
                    raise EncoderCacheLayoutError(
                        f"Identifier {identifier!r} has inconsistent row counts "
                        f"within request {req_id!r}: {state.num_embeds} and "
                        f"{num_embeds}."
                    )
                continue

            state = self.identifier_to_blocks.get(identifier)
            if state is not None:
                if state.num_embeds != num_embeds:
                    raise EncoderCacheLayoutError(
                        f"Identifier {identifier!r} has inconsistent row counts "
                        f"across active requests: {state.num_embeds} and "
                        f"{num_embeds}."
                    )
            else:
                state = IdentifierState(
                    num_embeds=num_embeds,
                    chunk_ids=self._make_chunk_ids(identifier, num_embeds),
                )
            request_states[identifier] = state

        identifiers = tuple(request_states)
        self.req_to_identifiers[req_id] = identifiers
        for identifier, candidate in request_states.items():
            state = self.identifier_to_blocks.setdefault(identifier, candidate)
            state.request_ref_count += 1
        return True

    def has_cache_item(self, identifier: str) -> bool:
        if not self.is_consumer:
            return False
        state = self.identifier_to_blocks.get(identifier)
        if state is None:
            return False
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        found = self.store.lookup(list(state.chunk_ids))
        hit = len(found) == len(state.chunk_ids) and all(bool(item) for item in found)
        if hit:
            self.step_verified_hits.add(identifier)
        return hit

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        identifier = request.mm_features[index].identifier
        if not self.is_consumer or identifier not in self.step_verified_hits:
            return
        if identifier in self.pending_loads:
            return
        state = self.identifier_to_blocks.get(identifier)
        if state is None:
            raise UCMEncoderCacheError(
                identifier,
                "Missing scheduler state for verified encoder cache item",
            )
        self.pending_loads[identifier] = ECLoadSpec(
            num_embeds=state.num_embeds,
            chunk_ids=state.chunk_ids,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> UCMECConnectorMetadata:
        del scheduler_output
        metadata = UCMECConnectorMetadata(loads=self.pending_loads)
        self.pending_loads = {}
        self.step_verified_hits.clear()
        return metadata

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        identifiers = self.req_to_identifiers.pop(request.request_id, None)
        if identifiers is None:
            return False, None

        for identifier in identifiers:
            state = self.identifier_to_blocks.get(identifier)
            if state is None:
                continue
            state.request_ref_count -= 1
            if state.request_ref_count <= 0:
                del self.identifier_to_blocks[identifier]
        return False, None

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        del connector_output

    def has_pending_push_work(self) -> bool:
        return False

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMECConnectorMetadata):
            raise TypeError(
                "UCMECConnector expected UCMECConnectorMetadata, got "
                f"{type(metadata).__name__}."
            )
        if self.device is None:
            raise RuntimeError("Worker accelerator device is not initialized.")
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        rows = self.layout.rows_per_chunk
        width = self.layout.width
        for identifier, spec in metadata.loads.items():
            if identifier in encoder_cache:
                continue
            expected_chunks = (
                spec.num_embeds + self.layout.rows_per_chunk - 1
            ) // self.layout.rows_per_chunk
            if spec.num_embeds <= 0 or len(spec.chunk_ids) != expected_chunks:
                raise EncoderCacheLayoutError(
                    f"Invalid load metadata for {identifier!r}: "
                    f"num_embeds={spec.num_embeds}, chunks={len(spec.chunk_ids)}, "
                    f"expected_chunks={expected_chunks}."
                )

            storage = torch.empty(
                (expected_chunks, rows, width),
                dtype=self.layout.dtype,
                device=self.device,
            )
            dst_chunks = [[storage[index]] for index in range(expected_chunks)]
            try:
                task = self.store.load(
                    list(spec.chunk_ids),
                    [0] * expected_chunks,
                    dst_chunks,
                )
                self.store.wait(task)
            except Exception as exc:
                encoder_cache.pop(identifier, None)
                raise UCMEncoderCacheError(
                    identifier, "Failed to load encoder cache item"
                ) from exc

            encoder_cache[identifier] = storage.view(-1, width)[: spec.num_embeds]

    def _validate_encoder_tensor(
        self,
        identifier: str,
        tensor: torch.Tensor,
    ) -> None:
        if tensor.ndim != 2:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} must be 2-D, got shape "
                f"{tuple(tensor.shape)}."
            )
        if int(tensor.shape[0]) <= 0:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} must contain at least one row."
            )
        if int(tensor.shape[1]) != self.layout.width:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} has width {tensor.shape[1]}, "
                f"expected {self.layout.width}."
            )
        if tensor.dtype != self.layout.dtype:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} has dtype {tensor.dtype}, "
                f"expected {self.layout.dtype}."
            )
        if not tensor.is_contiguous():
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} must be contiguous."
            )

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not self.is_producer or not self.is_save_rank:
            return
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        tensor = encoder_cache[mm_hash]
        self._validate_encoder_tensor(mm_hash, tensor)
        num_embeds = int(tensor.shape[0])
        rows = self.layout.rows_per_chunk
        width = self.layout.width
        chunk_ids = self._make_chunk_ids(mm_hash, num_embeds)

        num_full_chunks, tail_rows = divmod(num_embeds, rows)
        try:
            src_chunks = [
                [tensor[index * rows : (index + 1) * rows]]
                for index in range(num_full_chunks)
            ]
            tail: torch.Tensor | None = None
            if tail_rows:
                tail = torch.zeros(
                    (rows, width),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tail[:tail_rows].copy_(tensor[num_full_chunks * rows :])
                src_chunks.append([tail])

            task = self.store.dump(
                list(chunk_ids),
                [0] * len(chunk_ids),
                src_chunks,
            )
            self.store.wait(task)
        except Exception:
            logger.exception("Failed to dump EC item %s", mm_hash)

    def register_caches(
        self,
        ec_caches: dict[str, torch.Tensor],
    ) -> None:
        del ec_caches

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        return None, None

    def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
        return None

    def shutdown(self) -> None:
        self.pending_loads.clear()
        self.step_verified_hits.clear()
        self.req_to_identifiers.clear()
        self.identifier_to_blocks.clear()
        self.clear_connector_metadata()

        store = self.store
        self.store = None
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()


__all__ = [
    "ECLoadSpec",
    "EncoderCacheLayout",
    "EncoderCacheLayoutError",
    "IdentifierState",
    "UCMECConnector",
    "UCMECConnectorMetadata",
    "UCMEncoderCacheError",
    "create_ucm_ec_store",
    "make_chunk_ids",
    "resolve_cache_namespace",
    "resolve_encoder_cache_layout",
]

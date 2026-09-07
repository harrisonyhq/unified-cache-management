"""Configuration and layout tests; no accelerator or Store I/O is needed."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml

from ucm.integration.vllm import ucm_ec_connector as ec
from ucm.utils import Config


class ECConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ucm_connector_name": "CustomRegisteredStore",
            "ucm_connector_config": {"storage_backends": "/first:/second"},
            "encoder_cache_config": {"chunk_size": 3},
        }
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="org/model",
                dtype=torch.bfloat16,
                hf_config=None,
                multimodal_config=None,
                revision=None,
                code_revision=None,
                tokenizer_revision=None,
                get_inputs_embeds_size=lambda: 5120,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_rank=0,
                tensor_parallel_size=4,
            ),
            instance_id="instance",
            ec_transfer_config=SimpleNamespace(engine_id="engine"),
        )

    def layout(self, config=None):
        return ec.resolve_encoder_cache_layout(
            self.vllm_config,
            self.config["encoder_cache_config"] if config is None else config,
        )

    def create_store(self, role=ec.ECConnectorRole.SCHEDULER):
        with patch.object(ec.UcmConnectorFactoryV1, "create_connector") as factory:
            result = ec.create_ucm_ec_store(
                vllm_config=self.vllm_config,
                role=role,
                layout=self.layout(),
                ec_config=self.config,
                device_id=-1 if role == ec.ECConnectorRole.SCHEDULER else 0,
            )
        self.assertIs(result, factory.return_value)
        name, config = factory.call_args.args
        self.assertEqual(name, "CustomRegisteredStore")
        return config

    def test_ec_yaml_loads_without_kv_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ec.yaml"
            path.write_text(yaml.safe_dump({"ucm_ec_connector": self.config}))
            transfer = SimpleNamespace(
                ec_connector_extra_config={"UCM_CONFIG_FILE": str(path)}
            )
            self.assertEqual(Config.load_ec_config(transfer), self.config)

    def test_invalid_ec_yaml_fails_at_loading_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ec.yaml"
            transfer = SimpleNamespace(
                ec_connector_extra_config={"UCM_CONFIG_FILE": str(path)}
            )
            for content in ("", "[", "[]", "ucm_ec_connector: []"):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        Config.load_ec_config(transfer)

    def test_existing_kv_config_entrypoints_are_preserved(self):
        self.assertEqual(
            Config(None).get_config(), {"ucm_connector_name": "UcmNfsStore"}
        )
        inline = {"ucm_connectors": [{"ucm_connector_name": "ExistingStore"}]}
        self.assertEqual(
            Config(SimpleNamespace(kv_connector_extra_config=inline)).get_config(),
            inline,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kv.yaml"
            path.write_text(yaml.safe_dump(inline))
            transfer = SimpleNamespace(
                kv_connector_extra_config={"UCM_CONFIG_FILE": str(path)}
            )
            self.assertEqual(Config(transfer).get_config(), inline)

    def test_chunk_size_preserves_rows_across_widths_and_dtypes(self):
        for width, dtype, size in ((4096, torch.bfloat16, 2), (5120, torch.float32, 4)):
            with self.subTest(width=width, dtype=dtype):
                self.vllm_config.model_config.dtype = dtype
                self.vllm_config.model_config.get_inputs_embeds_size = lambda: width
                layout = self.layout()
                self.assertEqual(layout.rows_per_chunk, 3)
                self.assertEqual(layout.chunk_bytes, 3 * width * size)

    def test_invalid_chunk_rows_are_rejected(self):
        for rows in (0, -1, 1.5, "128", True, None):
            with self.subTest(rows=rows), self.assertRaises(ec.EncoderCacheLayoutError):
                self.layout({"chunk_size": rows})

    def test_explicit_ec_width_does_not_require_generic_input_width(self):
        model = self.vllm_config.model_config
        model.get_inputs_embeds_size = Mock(side_effect=AssertionError("not needed"))
        layout = self.layout({"chunk_size": 128, "encoder_cache_hidden_dim": 4096})
        self.assertEqual(layout.width, 4096)
        self.assertEqual(layout.chunk_bytes, 128 * 4096 * 2)
        model.get_inputs_embeds_size.assert_not_called()

    def test_deepstack_width_uses_structure_independently_of_model_name(self):
        model = self.vllm_config.model_config
        model.hf_config = SimpleNamespace(vision_config=SimpleNamespace(
            out_hidden_size=1024, deepstack_visual_indexes=[1, 3, 5],
        ))
        model.get_inputs_embeds_size = Mock(side_effect=AssertionError("not needed"))
        for name in ("org/renamed-model", "/local/checkpoint"):
            with self.subTest(name=name):
                model.model = name
                self.assertEqual(self.layout().width, 4096)

    def test_empty_deepstack_uses_output_width(self):
        self.vllm_config.model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                out_hidden_size=1024, deepstack_visual_indexes=[],
            ),
        )
        self.assertEqual(self.layout().width, 1024)

    def test_incomplete_vision_layout_uses_vllm_input_width(self):
        self.vllm_config.model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(out_hidden_size=1024),
        )
        self.assertEqual(self.layout().width, 5120)

    def test_nonpositive_explicit_width_is_rejected(self):
        for width in (0, -1):
            with (
                self.subTest(width=width),
                self.assertRaises(ec.EncoderCacheLayoutError),
            ):
                self.layout({"chunk_size": 128, "encoder_cache_hidden_dim": width})

    def test_pruning_is_rejected_without_an_enable_switch(self):
        self.vllm_config.model_config.multimodal_config = SimpleNamespace(
            is_multimodal_pruning_enabled=lambda: True,
        )
        with (
            patch.object(ec.ECConnectorBase, "__init__", return_value=None),
            patch.object(ec, "create_ucm_ec_store") as create_store,
            self.assertRaises(ec.EncoderCacheLayoutError),
        ):
            ec.UCMECConnector(self.vllm_config, ec.ECConnectorRole.SCHEDULER)
        create_store.assert_not_called()

    def test_cache_namespace_preserves_identity_hash_input(self):
        model = self.vllm_config.model_config
        model.revision = "weights-revision"
        model.code_revision = "code-revision"
        model.tokenizer_revision = "tokenizer-revision"
        model.hf_config = SimpleNamespace(
            _commit_hash="commit", architectures=["CustomArchitecture"],
        )
        model.multimodal_config = SimpleNamespace(compute_hash=lambda: "mm-hash")
        hasher = Mock(return_value=b"namespace")
        self.assertEqual(
            ec.resolve_cache_namespace(self.vllm_config, {}, hasher), b"namespace"
        )
        hasher.assert_called_once_with((
            "ucm-ec-namespace-v1",
            ("inferred", (
                "org/model", "weights-revision", "code-revision",
                "tokenizer-revision", "commit", ("CustomArchitecture",), "mm-hash",
            )),
        ))

    def test_explicit_namespace_bypasses_model_identity(self):
        hasher = Mock(return_value=b"namespace")
        ec.resolve_cache_namespace(
            SimpleNamespace(), {"cache_namespace": "shared"}, hasher
        )
        hasher.assert_called_once_with((
            "ucm-ec-namespace-v1", ("explicit", "shared"),
        ))

    def test_store_overrides_do_not_mutate_yaml_and_preserve_backend_string(self):
        self.config["ucm_connector_config"].update(
            share_buffer_enable=False,
            local_rank_size=99,
        )
        original = copy.deepcopy(self.config)
        config = self.create_store()
        self.assertEqual(self.config, original)
        self.assertEqual(config["storage_backends"], "/first:/second")
        self.assertTrue(config["share_buffer_enable"])
        self.assertEqual(config["local_rank_size"], 4)
        self.assertEqual(config["tensor_size_list"], [3 * 5120 * 2])
        self.assertNotIn("gpu_kv_buffer_addrs", config)
        self.assertNotIn("gpu_kv_buffer_sizes", config)

    def test_gdr_is_disabled_and_warns_only_when_configured(self):
        with patch.object(ec.logger, "warning") as warning:
            self.assertFalse(self.create_store()["use_gdr"])
            warning.assert_not_called()
        for value in (True, False):
            self.config["ucm_connector_config"]["use_gdr"] = value
            with (
                self.subTest(value=value),
                patch.object(ec.logger, "warning") as warning,
            ):
                self.assertFalse(self.create_store()["use_gdr"])
                warning.assert_called_once()

    def test_gc_owner_overrides_user_flag_and_passes_capacity_to_store(self):
        for role in ec.ECConnectorRole:
            for dp_rank in (0, 1):
                for capacity in (0, 10240):
                    with self.subTest(role=role, dp_rank=dp_rank, capacity=capacity):
                        self.vllm_config.parallel_config.data_parallel_rank = dp_rank
                        owner = role == ec.ECConnectorRole.SCHEDULER and dp_rank == 0
                        self.config["ucm_connector_config"].update(
                            posix_gc_enable=not owner, posix_capacity_gb=capacity,
                        )
                        config = self.create_store(role)
                        self.assertEqual(config["posix_gc_enable"], owner)
                        self.assertEqual(config["posix_capacity_gb"], capacity)

    def test_store_namespace_default_fallback_and_override(self):
        self.assertEqual(self.create_store()["unique_id"], "instance.ec.dp0")
        self.vllm_config.instance_id = ""
        self.assertEqual(self.create_store()["unique_id"], "engine.ec.dp0")
        self.config["encoder_cache_config"]["store_unique_id"] = "explicit"
        self.assertEqual(self.create_store()["unique_id"], "explicit")


if __name__ == "__main__":
    unittest.main()

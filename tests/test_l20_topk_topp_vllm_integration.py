import importlib.util
from pathlib import Path


def load_installer():
    path = Path("integrations/vllm/install_l20_topk_topp_sampler.py")
    spec = importlib.util.spec_from_file_location("install_l20_topk_topp_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_l20_topk_topp_installer_patches_vllm_sampler_points():
    module = load_installer()
    topk_source = """
from vllm.triton_utils import HAS_TRITON
def flashinfer_sample(logits, k, p, generators={}):
    assert not (k is None and p is None)
    if k is None:
        return None
class TopKTopPSampler:
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return logits, None
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""
    metadata_source = """
class SamplingMetadata:
    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
"""
    active_sampler_source = """
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
"""
    gpu_input_batch_source = """
import numpy as np
import torch
        self.top_k_reqs: set[str] = set()

        # Frequency penalty related data structures
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            pass
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
        return SamplingMetadata(
            temperature=temperature,
"""
    gpu_model_runner_source = """
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
"""
    worker_source = """
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
)
def sample():
        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
"""

    patched_topk = module.patch_topk_topp_sampler(topk_source)
    patched_metadata = module.patch_sampling_metadata(metadata_source)
    patched_active_sampler = module.patch_active_sampler(active_sampler_source)
    patched_gpu_input_batch = module.patch_gpu_input_batch(gpu_input_batch_source)
    patched_gpu_model_runner = module.patch_gpu_model_runner(gpu_model_runner_source)
    patched_worker = module.patch_worker_sampler(worker_source)

    assert "maybe_l20_topk_topp_sample" in patched_topk
    assert "l20_expanded_idx_mapping: torch.Tensor | None = None" in patched_topk
    assert "l20_seeds: torch.Tensor | None = None" in patched_topk
    assert "l20_positions: torch.Tensor | None = None" in patched_topk
    assert "l20_history_tokens: torch.Tensor | None = None" in patched_topk
    assert "l20_history_lengths: torch.Tensor | None = None" in patched_topk
    assert "l20_defer_penalties: bool = False" in patched_topk
    assert "expanded_idx_mapping=l20_expanded_idx_mapping" in patched_topk
    assert "history_tokens=l20_history_tokens" in patched_topk
    assert "defer_penalties=l20_defer_penalties" in patched_topk
    assert "return l20_sampled, None" in patched_topk
    assert "l20_expanded_idx_mapping: torch.Tensor | None" in patched_metadata
    assert "l20_seeds: torch.Tensor | None" in patched_metadata
    assert "l20_positions: torch.Tensor | None" in patched_metadata
    assert "l20_history_tokens: torch.Tensor | None" in patched_metadata
    assert "l20_history_lengths: torch.Tensor | None" in patched_metadata
    assert "l20_defer_penalties: bool" in patched_metadata
    assert "sampling_metadata.l20_expanded_idx_mapping" in patched_active_sampler
    assert "sampling_metadata.max_num_logprobs is None" in patched_active_sampler
    assert "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS" in patched_active_sampler
    assert "if l20_allow_logprobs" in patched_active_sampler
    assert "sampling_metadata.l20_seeds" in patched_active_sampler
    assert "sampling_metadata.l20_positions" in patched_active_sampler
    assert "sampling_metadata.l20_history_tokens" in patched_active_sampler
    assert "sampling_metadata.l20_defer_penalties" in patched_active_sampler
    assert "sampling_metadata.frequency_penalties" in patched_active_sampler
    assert 'getattr(sampling_metadata, "l20_defer_penalties", False)' in patched_active_sampler
    assert "self.l20_sampler_seeds" in patched_gpu_input_batch
    assert 'globals().get("PIN_MEMORY"' in patched_gpu_input_batch
    assert "self.l20_sampler_positions" in patched_gpu_input_batch
    assert "self.l20_sampler_positions_cpu_tensor" in patched_gpu_input_batch
    assert "self.l20_sampler_positions_cpu[:num_reqs]" in patched_gpu_input_batch
    assert "self.l20_sampler_indices[:num_reqs]" in patched_gpu_input_batch
    assert "import os" in patched_gpu_input_batch
    assert "VLLM_L20_TOPK_TOPP_DEFER_PENALTIES" in patched_gpu_input_batch
    assert "self.max_num_logprobs is None" in patched_gpu_input_batch
    assert "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS" in patched_gpu_input_batch
    assert "and l20_allow_logprobs" in patched_gpu_input_batch
    assert "l20_history_cpu = torch.full" in patched_gpu_input_batch
    assert "l20_defer_penalties = True" in patched_gpu_input_batch
    assert "l20_history_tokens=l20_history_tokens" in patched_gpu_input_batch
    assert "l20_expanded_idx_mapping=torch.arange" in patched_gpu_model_runner
    assert "maybe_l20_topk_topp_sample" in patched_worker
    assert "expanded_idx_mapping=expanded_idx_mapping" in patched_worker
    assert "seeds=self.sampling_states.seeds.gpu" in patched_worker
    assert "positions=pos" in patched_worker
    assert "top_k_value=int(l20_top_k_values[0])" in patched_worker
    assert "sampled = l20_sampled.to(torch.int64)" in patched_worker
    assert (
        "sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)"
        in patched_worker
    )
    assert "required: bool = True" in Path(
        "integrations/vllm/install_l20_topk_topp_sampler.py"
    ).read_text()
    assert "required=False" in Path(
        "integrations/vllm/install_l20_topk_topp_sampler.py"
    ).read_text()


def test_l20_topk_topp_helper_uses_vllm_rng_state():
    source = Path("integrations/vllm/l20_topk_topp_sampling.py").read_text()

    assert "VLLM_L20_TOPK_TOPP_SAMPLER" in source
    assert "VLLM_L20_TOPK_TOPP_SAMPLER_TRACE" in source
    assert "should_prefer_l20_topk_topp_sampling" in source
    assert "topk_topp_sample_with_vllm_rng_out" in source
    assert "topk_topp_sparse_penalty_sample_with_vllm_rng_out" in source
    assert "history_tokens: torch.Tensor | None = None" in source
    assert "history_lengths: torch.Tensor | None = None" in source
    assert "defer_penalties: bool = False" in source
    assert "missing_sparse_penalty_state" in source
    assert "unsafe deferred L20 top-k/top-p penalties fallback" in source
    assert "sparse_penalty" in source
    assert "per_request_generators" in source
    assert "missing_vllm_rng_state" in source
    assert "torch.rand" not in source


def test_sparse_sampling_ab_runner_detects_log_request_flag():
    source = Path("scripts/run_vllm_a100_flashinfer_sparse_sampling_ab.sh").read_text()

    assert "detect_no_log_requests_arg" in source
    assert "VLLM_LOG_REQUESTS_ARG" in source
    assert "--no-enable-log-requests" in source
    assert "--disable-log-requests" in source
    assert 'server_args+=("$no_log_requests_arg")' in source


def test_l20_topk_topp_installer_patches_forward_level_penalty_guard():
    module = load_installer()
    active_sampler_source = """
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
"""

    patched = module.patch_active_sampler(active_sampler_source)

    assert "sampling_metadata.l20_expanded_idx_mapping" in patched
    assert "sampling_metadata.max_num_logprobs is None" in patched
    assert "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS" in patched
    assert "sampling_metadata.l20_history_tokens" in patched
    assert 'getattr(sampling_metadata, "l20_defer_penalties", False)' in patched
    assert "if not getattr" in patched
    assert "logits = self.apply_penalties(logits, sampling_metadata)" in patched


def test_l20_topk_topp_installer_patches_v010_topk_sampler_shape():
    module = load_installer()
    topk_source = """
from typing import Optional

import torch
from vllm.platforms import current_platform
class TopKTopPSampler:
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return logits, None
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""

    patched = module.patch_topk_topp_sampler(topk_source)

    assert "maybe_l20_topk_topp_sample" in patched
    assert "**_: object" in patched
    assert "l20_expanded_idx_mapping: Optional[torch.Tensor] = None" in patched
    assert "l20_repetition_penalties: Optional[torch.Tensor] = None" in patched
    assert "contiguous_logits = logits.contiguous()" in patched
    assert "return l20_sampled, None" in patched


def test_l20_topk_topp_installer_is_idempotent_for_v010_topk_sampler():
    module = load_installer()
    topk_source = """
from typing import Optional

import torch
from vllm.platforms import current_platform
class TopKTopPSampler:
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return logits, None
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""

    patched_once = module.patch_topk_topp_sampler(topk_source)
    patched_twice = module.patch_topk_topp_sampler(patched_once)

    assert patched_twice == patched_once


def test_l20_topk_topp_installer_upgrades_old_active_sampler_patch():
    module = load_installer()
    source = module.SAMPLER_TOPK_CALL_PATCHED_NO_LOGPROBS_GATE + """
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
"""

    patched = module.patch_active_sampler(source)
    patched_twice = module.patch_active_sampler(patched)

    assert patched_twice == patched
    assert "sampling_metadata.max_num_logprobs is None" in patched
    assert "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS" in patched
    assert "else False" in patched
    assert "l20_defer_penalties=sampling_metadata.l20_defer_penalties" not in patched


def test_l20_topk_topp_installer_forces_forward_cuda_without_flashinfer():
    module = load_installer()
    topk_source = """
from typing import Optional

import torch
from vllm.platforms import current_platform
class TopKTopPSampler:
    def __init__(self, logprobs_mode):
        self.logprobs_mode = logprobs_mode
        # flashinfer optimization does not apply if intermediate
        # logprobs/logits after top_k/top_p need to be returned
        if logprobs_mode not in (LogprobsMode.PROCESSED_LOGITS,
                                 LogprobsMode.PROCESSED_LOGPROBS
                                 ) and current_platform.is_cuda():
            self.forward = self.forward_cuda
    def forward_native(self, logits, generators, k, p):
        return logits, None
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""

    patched = module.patch_topk_topp_sampler(topk_source)

    assert "VLLM_L20_TOPK_TOPP_SAMPLER" in patched
    assert "self.forward = self.forward_cuda" in patched
    assert '"is_flashinfer_available" in globals()' in patched
    assert "return self.forward_native(logits, generators, k, p)" in patched


def test_l20_topk_topp_installer_patches_v010_gpu_input_batch_shape():
    module = load_installer()
    gpu_input_batch_source = """
import numpy as np
import torch
        self.top_k_reqs: set[str] = set()

        # IDs of requests which do not support spec decoding
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[
                req_index] = sampling_params.frequency_penalty
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            pass
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
        return SamplingMetadata(
            temperature=temperature,
"""

    patched = module.patch_gpu_input_batch(gpu_input_batch_source)

    assert "import os" in patched
    assert "self.l20_sampler_seeds" in patched
    assert "self.l20_sampler_indices" in patched
    assert "self.l20_sampler_positions_cpu_tensor" in patched
    assert "self.l20_sampler_seeds_cpu[req_index] = l20_seed" in patched
    assert "l20_history_tokens=l20_history_tokens" in patched
    assert "l20_defer_penalties=l20_defer_penalties" in patched
    assert "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS" in patched

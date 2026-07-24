#!/usr/bin/env python3
"""Install the opt-in L20 top-k/top-p sampler hook into a vLLM checkout."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path


IMPORT_LINE = (
    "import os\n"
    "from vllm.v1.sample.ops.l20_topk_topp_sampling import "
    "maybe_l20_topk_topp_sample\n"
)
ALLOW_LOGPROBS_ENV = "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS"
PIN_MEMORY_EXPR = (
    'globals().get("PIN_MEMORY", locals().get("pin_memory", '
    'getattr(self, "pin_memory", False)))'
)

TOPK_IMPORT_MARKER = "from vllm.triton_utils import HAS_TRITON\n"
TOPK_IMPORT_MARKER_V010 = "from vllm.platforms import current_platform\n"
FLASHINFER_PATCH_POINT = """    assert not (k is None and p is None)
    if k is None:
"""
FLASHINFER_PATCHED = """    assert not (k is None and p is None)
    if k is None:
"""

TOPK_FORWARD_SIGNATURE = """    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""
TOPK_FORWARD_SIGNATURE_PATCHED = """    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        *,
        l20_expanded_idx_mapping: torch.Tensor | None = None,
        l20_seeds: torch.Tensor | None = None,
        l20_positions: torch.Tensor | None = None,
        l20_history_tokens: torch.Tensor | None = None,
        l20_history_lengths: torch.Tensor | None = None,
        l20_defer_penalties: bool = False,
        l20_frequency_penalties: torch.Tensor | None = None,
        l20_presence_penalties: torch.Tensor | None = None,
        l20_repetition_penalties: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""
TOPK_FORWARD_SIGNATURE_OPTIONAL = """    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
"""
TOPK_FORWARD_SIGNATURE_OPTIONAL_PATCHED = """    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
        *,
        l20_expanded_idx_mapping: Optional[torch.Tensor] = None,
        l20_seeds: Optional[torch.Tensor] = None,
        l20_positions: Optional[torch.Tensor] = None,
        l20_history_tokens: Optional[torch.Tensor] = None,
        l20_history_lengths: Optional[torch.Tensor] = None,
        l20_defer_penalties: bool = False,
        l20_frequency_penalties: Optional[torch.Tensor] = None,
        l20_presence_penalties: Optional[torch.Tensor] = None,
        l20_repetition_penalties: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
"""
TOPK_NATIVE_SIGNATURE = """    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""
TOPK_NATIVE_SIGNATURE_PATCHED = """    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""
TOPK_NATIVE_SIGNATURE_OPTIONAL = """    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
"""
TOPK_NATIVE_SIGNATURE_OPTIONAL_PATCHED = """    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
        **_: object,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
"""
TOPK_FORCE_FORWARD_MARKER = """        self.logprobs_mode = logprobs_mode
        # flashinfer optimization does not apply if intermediate
        # logprobs/logits after top_k/top_p need to be returned
        if logprobs_mode not in (LogprobsMode.PROCESSED_LOGITS,
                                 LogprobsMode.PROCESSED_LOGPROBS
                                 ) and current_platform.is_cuda():
"""
TOPK_FORCE_FORWARD_PATCHED = """        self.logprobs_mode = logprobs_mode
        if (
            os.environ.get("VLLM_L20_TOPK_TOPP_SAMPLER", "0").lower()
            in {"1", "true", "yes", "on"}
            and logprobs_mode not in (LogprobsMode.PROCESSED_LOGITS,
                                      LogprobsMode.PROCESSED_LOGPROBS)
            and current_platform.is_cuda()
        ):
            logger.info_once(
                "Using experimental L20 top-p & top-k sampler hook.")
            self.forward = self.forward_cuda
        # flashinfer optimization does not apply if intermediate
        # logprobs/logits after top_k/top_p need to be returned
        elif logprobs_mode not in (LogprobsMode.PROCESSED_LOGITS,
                                   LogprobsMode.PROCESSED_LOGPROBS
                                   ) and current_platform.is_cuda():
"""
TOPK_FLASHINFER_RETURN = """        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""
TOPK_FLASHINFER_RETURN_PATCHED = """        contiguous_logits = logits.contiguous()
        l20_sampled = maybe_l20_topk_topp_sample(
            contiguous_logits,
            k,
            p,
            generators,
            expanded_idx_mapping=l20_expanded_idx_mapping,
            seeds=l20_seeds,
            positions=l20_positions,
            history_tokens=l20_history_tokens,
            history_lengths=l20_history_lengths,
            frequency_penalties=l20_frequency_penalties,
            presence_penalties=l20_presence_penalties,
            repetition_penalties=l20_repetition_penalties,
            defer_penalties=False,
        )
        if l20_sampled is not None:
            return l20_sampled, None
        if "is_flashinfer_available" in globals() and not is_flashinfer_available:
            return self.forward_native(logits, generators, k, p)
        return flashinfer_sample(contiguous_logits, k, p, generators), None
"""
TOPK_FLASHINFER_RETURN_PATCHED_LEGACY_DEFERRED = """        contiguous_logits = logits.contiguous()
        l20_sampled = maybe_l20_topk_topp_sample(
            contiguous_logits,
            k,
            p,
            generators,
            expanded_idx_mapping=l20_expanded_idx_mapping,
            seeds=l20_seeds,
            positions=l20_positions,
            history_tokens=l20_history_tokens,
            history_lengths=l20_history_lengths,
            frequency_penalties=l20_frequency_penalties,
            presence_penalties=l20_presence_penalties,
            repetition_penalties=l20_repetition_penalties,
            defer_penalties=l20_defer_penalties,
        )
        if l20_sampled is not None:
            return l20_sampled, None
        if "is_flashinfer_available" in globals() and not is_flashinfer_available:
            return self.forward_native(logits, generators, k, p)
        return flashinfer_sample(contiguous_logits, k, p, generators), None
"""

METADATA_GENERATORS = """    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
"""
METADATA_GENERATORS_PATCHED = """    generators: dict[int, torch.Generator]

    # L20 experimental sampler state. These are None for upstream/default paths.
    l20_expanded_idx_mapping: torch.Tensor | None
    l20_seeds: torch.Tensor | None
    l20_positions: torch.Tensor | None
    l20_history_tokens: torch.Tensor | None
    l20_history_lengths: torch.Tensor | None
    l20_defer_penalties: bool

    # None means no logprobs, 0 means sampled token logprobs only
"""

SAMPLER_TOPK_CALL = """        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
"""
SAMPLER_TOPK_CALL_PATCHED_NO_LOGPROBS_GATE = """        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            l20_expanded_idx_mapping=sampling_metadata.l20_expanded_idx_mapping,
            l20_seeds=sampling_metadata.l20_seeds,
            l20_positions=sampling_metadata.l20_positions,
            l20_history_tokens=sampling_metadata.l20_history_tokens,
            l20_history_lengths=sampling_metadata.l20_history_lengths,
            l20_defer_penalties=False,
            l20_frequency_penalties=sampling_metadata.frequency_penalties,
            l20_presence_penalties=sampling_metadata.presence_penalties,
            l20_repetition_penalties=sampling_metadata.repetition_penalties,
        )
"""
SAMPLER_TOPK_CALL_PATCHED_NO_LOGPROBS_GATE_LEGACY_DEFERRED = """        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            l20_expanded_idx_mapping=sampling_metadata.l20_expanded_idx_mapping,
            l20_seeds=sampling_metadata.l20_seeds,
            l20_positions=sampling_metadata.l20_positions,
            l20_history_tokens=sampling_metadata.l20_history_tokens,
            l20_history_lengths=sampling_metadata.l20_history_lengths,
            l20_defer_penalties=sampling_metadata.l20_defer_penalties,
            l20_frequency_penalties=sampling_metadata.frequency_penalties,
            l20_presence_penalties=sampling_metadata.presence_penalties,
            l20_repetition_penalties=sampling_metadata.repetition_penalties,
        )
"""
SAMPLER_TOPK_CALL_PATCHED = """        l20_allow_logprobs = (
            sampling_metadata.max_num_logprobs is None
            or __import__("os").environ.get(
                "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS", "0"
            ).lower() in {"1", "true", "yes", "on"}
        )
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            l20_expanded_idx_mapping=(
                sampling_metadata.l20_expanded_idx_mapping
                if l20_allow_logprobs
                else None
            ),
            l20_seeds=(
                sampling_metadata.l20_seeds
                if l20_allow_logprobs
                else None
            ),
            l20_positions=(
                sampling_metadata.l20_positions
                if l20_allow_logprobs
                else None
            ),
            l20_history_tokens=(
                sampling_metadata.l20_history_tokens
                if l20_allow_logprobs
                else None
            ),
            l20_history_lengths=(
                sampling_metadata.l20_history_lengths
                if l20_allow_logprobs
                else None
            ),
            l20_defer_penalties=False,
            l20_frequency_penalties=sampling_metadata.frequency_penalties,
            l20_presence_penalties=sampling_metadata.presence_penalties,
            l20_repetition_penalties=sampling_metadata.repetition_penalties,
        )
"""
SAMPLER_TOPK_CALL_PATCHED_LEGACY_DEFERRED = """        l20_allow_logprobs = (
            sampling_metadata.max_num_logprobs is None
            or __import__("os").environ.get(
                "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS", "0"
            ).lower() in {"1", "true", "yes", "on"}
        )
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            l20_expanded_idx_mapping=(
                sampling_metadata.l20_expanded_idx_mapping
                if l20_allow_logprobs
                else None
            ),
            l20_seeds=(
                sampling_metadata.l20_seeds
                if l20_allow_logprobs
                else None
            ),
            l20_positions=(
                sampling_metadata.l20_positions
                if l20_allow_logprobs
                else None
            ),
            l20_history_tokens=(
                sampling_metadata.l20_history_tokens
                if l20_allow_logprobs
                else None
            ),
            l20_history_lengths=(
                sampling_metadata.l20_history_lengths
                if l20_allow_logprobs
                else None
            ),
            l20_defer_penalties=(
                sampling_metadata.l20_defer_penalties
                if l20_allow_logprobs
                else False
            ),
            l20_frequency_penalties=sampling_metadata.frequency_penalties,
            l20_presence_penalties=sampling_metadata.presence_penalties,
            l20_repetition_penalties=sampling_metadata.repetition_penalties,
        )
"""
SAMPLER_APPLY_PENALTIES = """        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
"""
SAMPLER_APPLY_PENALTIES_PATCHED = """        if sampling_metadata.no_penalties:
            return logits
        if getattr(sampling_metadata, "l20_defer_penalties", False):
            return logits

        assert sampling_metadata.prompt_token_ids is not None
"""
SAMPLER_FORWARD_APPLY_PENALTIES = """        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)
"""
SAMPLER_FORWARD_APPLY_PENALTIES_PATCHED = """        # Apply penalties (e.g., min_tokens, freq_penalties).
        if not getattr(sampling_metadata, "l20_defer_penalties", False):
            logits = self.apply_penalties(logits, sampling_metadata)
"""

DUMMY_METADATA_MARKER = """            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
"""
DUMMY_METADATA_PATCHED = """            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            l20_expanded_idx_mapping=torch.arange(
                num_reqs, dtype=torch.int64, device=self.device
            ),
            l20_seeds=torch.full(
                (num_reqs,), 1, dtype=torch.int64, device=self.device
            ),
            l20_positions=torch.arange(
                num_reqs, dtype=torch.int64, device=self.device
            ),
            l20_history_tokens=None,
            l20_history_lengths=None,
            l20_defer_penalties=False,
"""

INPUT_BATCH_TOPK_REQS = """        self.top_k_reqs: set[str] = set()

        # Frequency penalty related data structures
"""
INPUT_BATCH_TOPK_REQS_V010 = """        self.top_k_reqs: set[str] = set()

        # IDs of requests which do not support spec decoding
"""
INPUT_BATCH_TOPK_REQS_PATCHED = f"""        self.top_k_reqs: set[str] = set()

        self.l20_sampler_seeds = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=device
        )
        self.l20_sampler_seeds_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=\"cpu\", pin_memory={PIN_MEMORY_EXPR}
        )
        self.l20_sampler_seeds_cpu = self.l20_sampler_seeds_cpu_tensor.numpy()
        self.l20_sampler_positions = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=device
        )
        self.l20_sampler_positions_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device="cpu", pin_memory={PIN_MEMORY_EXPR}
        )
        self.l20_sampler_positions_cpu = self.l20_sampler_positions_cpu_tensor.numpy()
        self.l20_sampler_indices = torch.arange(
            max_num_reqs, dtype=torch.int64, device=device
        )

        # Frequency penalty related data structures
"""
INPUT_BATCH_TOPK_REQS_V010_PATCHED = f"""        self.top_k_reqs: set[str] = set()

        self.l20_sampler_seeds = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=device
        )
        self.l20_sampler_seeds_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device="cpu", pin_memory={PIN_MEMORY_EXPR}
        )
        self.l20_sampler_seeds_cpu = self.l20_sampler_seeds_cpu_tensor.numpy()
        self.l20_sampler_positions = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=device
        )
        self.l20_sampler_positions_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device="cpu", pin_memory={PIN_MEMORY_EXPR}
        )
        self.l20_sampler_positions_cpu = self.l20_sampler_positions_cpu_tensor.numpy()
        self.l20_sampler_indices = torch.arange(
            max_num_reqs, dtype=torch.int64, device=device
        )

        # IDs of requests which do not support spec decoding
"""

INPUT_BATCH_TOPK_CPU = """            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
"""
INPUT_BATCH_TOPK_CPU_PATCHED = """            self.top_k_cpu[req_index] = top_k
            l20_seed = sampling_params.seed
            if l20_seed is None:
                l20_seed = np.random.randint(
                    np.iinfo(np.int64).min, np.iinfo(np.int64).max
                )
            self.l20_sampler_seeds_cpu[req_index] = l20_seed
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
"""
INPUT_BATCH_TOPK_CPU_V010 = """            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[
                req_index] = sampling_params.frequency_penalty
"""
INPUT_BATCH_TOPK_CPU_V010_PATCHED = """            self.top_k_cpu[req_index] = top_k
            l20_seed = sampling_params.seed
            if l20_seed is None:
                l20_seed = np.random.randint(
                    np.iinfo(np.int64).min, np.iinfo(np.int64).max
                )
            self.l20_sampler_seeds_cpu[req_index] = l20_seed
            self.frequency_penalties_cpu[
                req_index] = sampling_params.frequency_penalty
"""

INPUT_BATCH_COPY_TOPK = """        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
"""
INPUT_BATCH_COPY_TOPK_PATCHED = """        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)
        copy_slice(self.l20_sampler_seeds_cpu_tensor, self.l20_sampler_seeds, num_reqs)
        self.l20_sampler_positions_cpu[:num_reqs] = self.num_tokens_no_spec[:num_reqs]
        copy_slice(
            self.l20_sampler_positions_cpu_tensor, self.l20_sampler_positions, num_reqs
        )

        if not self.no_penalties:
"""

INPUT_BATCH_METADATA_GENERATORS = """            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
"""
INPUT_BATCH_METADATA_GENERATORS_PATCHED = """            generators=self.generators,
            l20_expanded_idx_mapping=self.l20_sampler_indices[:num_reqs],
            l20_seeds=self.l20_sampler_seeds[:num_reqs],
            l20_positions=self.l20_sampler_positions[:num_reqs],
            l20_history_tokens=l20_history_tokens,
            l20_history_lengths=l20_history_lengths,
            l20_defer_penalties=l20_defer_penalties,
            max_num_logprobs=self.max_num_logprobs,
"""

INPUT_BATCH_METADATA_RETURN = """        return SamplingMetadata(
            temperature=temperature,
"""
INPUT_BATCH_SPARSE_HISTORY_LEGACY_DEFERRED = f"""        l20_history_tokens = None
        l20_history_lengths = None
        l20_defer_penalties = False
        l20_allow_logprobs = (
            self.max_num_logprobs is None
            or os.environ.get("{ALLOW_LOGPROBS_ENV}", "0").lower()
            in {{"1", "true", "yes", "on"}}
        )
        if (
            os.environ.get("VLLM_L20_TOPK_TOPP_DEFER_PENALTIES", "0").lower() in {{"1", "true", "yes", "on"}}
            and not self.no_penalties
            and l20_allow_logprobs
            and num_reqs <= 4
        ):
            l20_max_history = 256
            l20_history_cpu = torch.full(
                (num_reqs, l20_max_history),
                self.vocab_size,
                dtype=torch.int64,
                device="cpu",
                pin_memory={PIN_MEMORY_EXPR},
            )
            l20_lengths_cpu = torch.zeros(
                (num_reqs,),
                dtype=torch.int32,
                device="cpu",
                pin_memory={PIN_MEMORY_EXPR},
            )
            for row in range(num_reqs):
                token_count = int(self.num_tokens_no_spec[row])
                start = max(0, token_count - l20_max_history)
                active = self.token_ids_cpu[row, start:token_count]
                length = min(int(active.shape[0]), l20_max_history)
                if length > 0:
                    l20_history_cpu[row, :length] = torch.as_tensor(
                        active[:length], dtype=torch.int64
                    )
                l20_lengths_cpu[row] = length
            l20_history_tokens = l20_history_cpu.to(
                device=self.device, non_blocking=True
            )
            l20_history_lengths = l20_lengths_cpu.to(
                device=self.device, non_blocking=True
            )
            l20_defer_penalties = True

        return SamplingMetadata(
            temperature=temperature,
"""
INPUT_BATCH_SPARSE_HISTORY = """        # Keep native vLLM penalties active. The sampler's final eligibility
        # decision happens later, so deferring penalties here could make an
        # otherwise valid fallback semantically unsafe.
        l20_history_tokens = None
        l20_history_lengths = None
        l20_defer_penalties = False

        return SamplingMetadata(
            temperature=temperature,
"""

WORKER_IMPORT_MARKER = "from vllm.v1.sample.ops.topk_topp_sampler import (\n"
WORKER_PATCH_POINT = """        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
"""
WORKER_PATCHED = """        if use_flashinfer:
            l20_top_k_values = self.sampling_states.top_k.np[idx_mapping_np]
            l20_top_p_values = self.sampling_states.top_p.np[idx_mapping_np]
            l20_top_k_uniform = bool((l20_top_k_values == l20_top_k_values[0]).all())
            l20_top_p_uniform = bool((l20_top_p_values == l20_top_p_values[0]).all())
            l20_sampled = None
            if l20_top_k_uniform and l20_top_p_uniform:
                l20_sampled = maybe_l20_topk_topp_sample(
                    processed_logits,
                    top_k,
                    top_p,
                    expanded_idx_mapping=expanded_idx_mapping,
                    seeds=self.sampling_states.seeds.gpu,
                    positions=pos,
                    top_k_value=int(l20_top_k_values[0]),
                    top_p_value=float(l20_top_p_values[0]),
                )
            if l20_sampled is not None:
                sampled = l20_sampled.to(torch.int64)
            else:
                sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="Path to a vLLM source checkout root. Defaults to imported package.",
    )
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def patch_topk_topp_sampler(source: str) -> str:
    if IMPORT_LINE in source:
        pass
    elif TOPK_IMPORT_MARKER in source:
        source = replace_once(
            source,
            TOPK_IMPORT_MARKER,
            TOPK_IMPORT_MARKER + IMPORT_LINE,
            "topk_topp_sampler import",
        )
    else:
        source = replace_once(
            source,
            TOPK_IMPORT_MARKER_V010,
            TOPK_IMPORT_MARKER_V010 + IMPORT_LINE,
            "topk_topp_sampler v0.10 import",
        )
    if FLASHINFER_PATCH_POINT in source:
        source = replace_once(
            source,
            FLASHINFER_PATCH_POINT,
            FLASHINFER_PATCHED,
            "flashinfer_sample hook",
        )
    if (
        TOPK_FORWARD_SIGNATURE in source
        or TOPK_FORWARD_SIGNATURE_PATCHED in source
    ):
        source = replace_once(
            source,
            TOPK_FORWARD_SIGNATURE,
            TOPK_FORWARD_SIGNATURE_PATCHED,
            "topk_topp forward_cuda signature",
        )
    else:
        source = replace_once(
            source,
            TOPK_FORWARD_SIGNATURE_OPTIONAL,
            TOPK_FORWARD_SIGNATURE_OPTIONAL_PATCHED,
            "topk_topp forward_cuda optional signature",
        )
    if TOPK_NATIVE_SIGNATURE in source or TOPK_NATIVE_SIGNATURE_PATCHED in source:
        source = replace_once(
            source,
            TOPK_NATIVE_SIGNATURE,
            TOPK_NATIVE_SIGNATURE_PATCHED,
            "topk_topp forward_native signature",
        )
    elif (
        TOPK_NATIVE_SIGNATURE_OPTIONAL in source
        or TOPK_NATIVE_SIGNATURE_OPTIONAL_PATCHED in source
    ):
        source = replace_once(
            source,
            TOPK_NATIVE_SIGNATURE_OPTIONAL,
            TOPK_NATIVE_SIGNATURE_OPTIONAL_PATCHED,
            "topk_topp forward_native optional signature",
        )
    if TOPK_FORCE_FORWARD_MARKER in source or TOPK_FORCE_FORWARD_PATCHED in source:
        source = replace_once(
            source,
            TOPK_FORCE_FORWARD_MARKER,
            TOPK_FORCE_FORWARD_PATCHED,
            "topk_topp force l20 forward_cuda",
        )
    source = source.replace(
        TOPK_FLASHINFER_RETURN_PATCHED_LEGACY_DEFERRED,
        TOPK_FLASHINFER_RETURN_PATCHED,
        1,
    )
    return replace_once(
        source,
        TOPK_FLASHINFER_RETURN,
        TOPK_FLASHINFER_RETURN_PATCHED,
        "topk_topp forward_cuda l20 hook",
    )


def patch_sampling_metadata(source: str) -> str:
    return replace_once(
        source,
        METADATA_GENERATORS,
        METADATA_GENERATORS_PATCHED,
        "sampling metadata l20 state",
    )


def patch_active_sampler(source: str) -> str:
    # Repair installations made by the earlier experimental deferred-penalty
    # patch. Native penalties must always execute before the optional sampler;
    # otherwise any later eligibility fallback can silently change semantics.
    source = source.replace(
        SAMPLER_APPLY_PENALTIES_PATCHED,
        SAMPLER_APPLY_PENALTIES,
        1,
    )
    source = source.replace(
        SAMPLER_FORWARD_APPLY_PENALTIES_PATCHED,
        SAMPLER_FORWARD_APPLY_PENALTIES,
        1,
    )
    source = source.replace(
        SAMPLER_TOPK_CALL_PATCHED_LEGACY_DEFERRED,
        SAMPLER_TOPK_CALL_PATCHED,
        1,
    )
    source = source.replace(
        SAMPLER_TOPK_CALL_PATCHED_NO_LOGPROBS_GATE_LEGACY_DEFERRED,
        SAMPLER_TOPK_CALL_PATCHED,
        1,
    )
    source = source.replace(
        SAMPLER_TOPK_CALL_PATCHED_NO_LOGPROBS_GATE,
        SAMPLER_TOPK_CALL_PATCHED,
        1,
    )
    source = replace_once(
        source,
        SAMPLER_TOPK_CALL,
        SAMPLER_TOPK_CALL_PATCHED,
        "active sampler topk_topp state pass",
    )
    return source


def patch_gpu_model_runner(source: str) -> str:
    return replace_once(
        source,
        DUMMY_METADATA_MARKER,
        DUMMY_METADATA_PATCHED,
        "dummy sampler metadata l20 state",
    )


def patch_gpu_input_batch(source: str) -> str:
    source = source.replace(
        "pin_memory=pin_memory",
        f"pin_memory={PIN_MEMORY_EXPR}",
    ).replace(
        "pin_memory=self.pin_memory",
        f"pin_memory={PIN_MEMORY_EXPR}",
    )
    if INPUT_BATCH_TOPK_REQS in source or INPUT_BATCH_TOPK_REQS_PATCHED in source:
        source = replace_once(
            source,
            INPUT_BATCH_TOPK_REQS,
            INPUT_BATCH_TOPK_REQS_PATCHED,
            "gpu input batch l20 buffers",
        )
    else:
        source = replace_once(
            source,
            INPUT_BATCH_TOPK_REQS_V010,
            INPUT_BATCH_TOPK_REQS_V010_PATCHED,
            "gpu input batch v0.10 l20 buffers",
        )
    if INPUT_BATCH_TOPK_CPU in source or INPUT_BATCH_TOPK_CPU_PATCHED in source:
        source = replace_once(
            source,
            INPUT_BATCH_TOPK_CPU,
            INPUT_BATCH_TOPK_CPU_PATCHED,
            "gpu input batch l20 seed init",
        )
    else:
        source = replace_once(
            source,
            INPUT_BATCH_TOPK_CPU_V010,
            INPUT_BATCH_TOPK_CPU_V010_PATCHED,
            "gpu input batch v0.10 l20 seed init",
        )
    source = replace_once(
        source,
        INPUT_BATCH_COPY_TOPK,
        INPUT_BATCH_COPY_TOPK_PATCHED,
        "gpu input batch l20 copy",
    )
    source = source.replace(
        INPUT_BATCH_SPARSE_HISTORY_LEGACY_DEFERRED,
        INPUT_BATCH_SPARSE_HISTORY,
        1,
    )
    source = replace_once(
        source,
        INPUT_BATCH_METADATA_RETURN,
        INPUT_BATCH_SPARSE_HISTORY,
        "gpu input batch sparse history metadata",
    )
    return replace_once(
        source,
        INPUT_BATCH_METADATA_GENERATORS,
        INPUT_BATCH_METADATA_GENERATORS_PATCHED,
        "gpu input batch metadata l20 state",
    )


def patch_worker_sampler(source: str) -> str:
    source = replace_once(
        source,
        WORKER_IMPORT_MARKER,
        WORKER_IMPORT_MARKER + "    maybe_l20_topk_topp_sample,\n",
        "worker sampler import",
    )
    return replace_once(
        source,
        WORKER_PATCH_POINT,
        WORKER_PATCHED,
        "worker native sampler hook",
    )


def _install_target(path: Path, patcher, *, required: bool = True) -> bool:
    if not path.exists():
        return False
    backup = path.with_suffix(".py.l20-topk-topp-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    try:
        patched = patcher(path.read_text(encoding="utf-8"))
    except RuntimeError:
        if required:
            raise
        return False
    path.write_text(patched, encoding="utf-8")
    return True


def _restore_target(path: Path) -> bool:
    backup = path.with_suffix(".py.l20-topk-topp-backup")
    if not backup.exists():
        return False
    shutil.copy2(backup, path)
    return True


def install(package: Path) -> None:
    helper = package / "v1" / "sample" / "ops" / "l20_topk_topp_sampling.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("l20_topk_topp_sampling.py"), helper)
    patched = [
        _install_target(
            package / "v1" / "sample" / "metadata.py",
            patch_sampling_metadata,
        ),
        _install_target(
            package / "v1" / "sample" / "sampler.py",
            patch_active_sampler,
        ),
        _install_target(
            package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
            patch_topk_topp_sampler,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu_input_batch.py",
            patch_gpu_input_batch,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu_model_runner.py",
            patch_gpu_model_runner,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
            patch_worker_sampler,
            required=False,
        ),
    ]
    if not any(patched):
        raise RuntimeError(f"missing supported vLLM sampler under: {package}")


def uninstall(package: Path) -> None:
    paths = [
        package / "v1" / "sample" / "metadata.py",
        package / "v1" / "sample" / "sampler.py",
        package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        package / "v1" / "worker" / "gpu_input_batch.py",
        package / "v1" / "worker" / "gpu_model_runner.py",
        package / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
    ]
    for path in paths:
        _restore_target(path)
    (package / "v1" / "sample" / "ops" / "l20_topk_topp_sampling.py").unlink(
        missing_ok=True
    )


def main() -> int:
    args = parse_args()
    package = resolve_package(args.vllm_source)
    if args.uninstall:
        uninstall(package)
    else:
        install(package)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

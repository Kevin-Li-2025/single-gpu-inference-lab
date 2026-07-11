#ifndef KEVIN_M4_Q4K_SME2_H
#define KEVIN_M4_Q4K_SME2_H

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "../repack.h"

static constexpr uint32_t KEVIN_M4_Q4K_SME2_MAGIC = 0x4b513453; // "KQ4S"
static constexpr uint32_t KEVIN_M4_Q4K_SME2_VERSION = 2;
static constexpr size_t KEVIN_M4_Q4K_SME2_ALIGN = 64;
static constexpr size_t KEVIN_M4_Q4K_SME2_BL = 32;
static constexpr size_t KEVIN_M4_Q4K_SME2_SOURCE_BLOCK = 18;
static constexpr size_t KEVIN_M4_Q4K_SME2_MAX_THREADS = 64;

struct kevin_m4_q4k_sme2_header {
    uint32_t magic;
    uint32_t version;
    uint64_t n;
    uint64_t k;
    uint64_t packed_offset;
    uint64_t packed_size;
    uint64_t correction_offset;
    uint64_t correction_size;
    uint64_t fallback_offset;
    uint64_t fallback_size;
};

static size_t kevin_m4_q4k_sme2_align_up(size_t value) {
    return (value + KEVIN_M4_Q4K_SME2_ALIGN - 1) & ~(KEVIN_M4_Q4K_SME2_ALIGN - 1);
}

static bool kevin_m4_q4k_sme2_enabled(void) {
    const char * value = getenv("GGML_M4_Q4K_SME2");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

static bool kevin_m4_q4k_sme2_tensor_role_enabled(const ggml_tensor * tensor) {
    const char * roles = getenv("GGML_M4_Q4K_SME2_TENSORS");
    const bool is_up = strstr(tensor->name, ".ffn_up.") != nullptr;
    const bool is_gate = strstr(tensor->name, ".ffn_gate.") != nullptr;
    const bool is_down = strstr(tensor->name, ".ffn_down.") != nullptr;
    if (roles == nullptr) {
        return is_down;
    }
    if (strcmp(roles, "all") == 0) {
        return true;
    }
    return (is_up && strstr(roles, "up") != nullptr) ||
           (is_gate && strstr(roles, "gate") != nullptr) ||
           (is_down && strstr(roles, "down") != nullptr);
}

static bool kevin_m4_q4k_sme2_tensor_eligible(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->type == GGML_TYPE_Q4_K &&
           tensor->ne[0] * tensor->ne[1] >= 8 * 1024 * 1024 &&
           tensor->ne[1] % 8 == 0 &&
           strstr(tensor->name, ".ffn_") != nullptr &&
           kevin_m4_q4k_sme2_tensor_role_enabled(tensor);
}

static bool kevin_m4_q4k_sme2_trace_enabled(void) {
    const char * value = getenv("GGML_M4_Q4K_SME2_TRACE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

static int kevin_m4_q4k_sme2_share_percent(void) {
    const char * value = getenv("GGML_M4_Q4K_SME2_SHARE_PERCENT");
    if (value == nullptr) {
        return 25;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed >= 5 && parsed <= 50
        ? static_cast<int>(parsed)
        : 25;
}

static bool kevin_m4_q4k_sme2_shared_q8_enabled(void) {
    const char * value = getenv("GGML_M4_Q4K_SME2_SHARED_Q8");
    return value == nullptr || (value[0] == '1' && value[1] == '\0');
}

static bool kevin_m4_q4k_sme2_parallel_correction_enabled(void) {
    const char * value = getenv("GGML_M4_Q4K_SME2_PARALLEL_CORRECTION");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

static void kevin_m4_q4k_sme2_trace_once(void) {
    static std::atomic_flag emitted = ATOMIC_FLAG_INIT;
    if (kevin_m4_q4k_sme2_trace_enabled() && !emitted.test_and_set(std::memory_order_relaxed)) {
        GGML_LOG_INFO("kevin_m4_q4k_sme2: persistent affine Q4_K SME2 decode path hit\n");
    }
}

static void kevin_m4_q4k_decode_scale_min(
        const uint8_t * packed, int group, uint8_t & scale, uint8_t & minimum) {
    if (group < 4) {
        scale = packed[group] & 63;
        minimum = packed[group + 4] & 63;
    } else {
        scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
        minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

static uint8_t kevin_m4_q4k_value(const block_q4_K & block, int group, int index) {
    const uint8_t packed = block.qs[(group / 2) * 32 + index];
    return (group & 1) != 0 ? packed >> 4 : packed & 0x0f;
}

static size_t kevin_m4_q4k_sme2_packed_size(size_t n, size_t k) {
    const size_t nr =
        kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
        n, k, nr, kr, KEVIN_M4_Q4K_SME2_BL);
}

static size_t kevin_m4_q4k_sme2_alloc_size(const ggml_tensor * tensor) {
    const size_t original_size = ggml_nbytes(tensor);
    if (!kevin_m4_q4k_sme2_enabled() || !kevin_m4_q4k_sme2_tensor_eligible(tensor) ||
        tensor->ne[0] % QK_K != 0) {
        return original_size;
    }
    const size_t n = tensor->ne[1];
    const size_t k = tensor->ne[0];
    const size_t header_offset = kevin_m4_q4k_sme2_align_up(original_size);
    const size_t packed_offset =
        kevin_m4_q4k_sme2_align_up(header_offset + sizeof(kevin_m4_q4k_sme2_header));
    const size_t correction_offset =
        kevin_m4_q4k_sme2_align_up(packed_offset + kevin_m4_q4k_sme2_packed_size(n, k));
    const size_t correction_size = n * (k / KEVIN_M4_Q4K_SME2_BL) * sizeof(float);
    const size_t fallback_offset =
        kevin_m4_q4k_sme2_align_up(correction_offset + correction_size);
    return fallback_offset + original_size;
}

static kevin_m4_q4k_sme2_header * kevin_m4_q4k_sme2_header_for(ggml_tensor * tensor) {
    uint8_t * base = static_cast<uint8_t *>(tensor->data);
    return reinterpret_cast<kevin_m4_q4k_sme2_header *>(
        base + kevin_m4_q4k_sme2_align_up(ggml_nbytes(tensor)));
}

static const kevin_m4_q4k_sme2_header * kevin_m4_q4k_sme2_header_for(const ggml_tensor * tensor) {
    const uint8_t * base = static_cast<const uint8_t *>(tensor->data);
    return reinterpret_cast<const kevin_m4_q4k_sme2_header *>(
        base + kevin_m4_q4k_sme2_align_up(ggml_nbytes(tensor)));
}

static bool kevin_m4_q4k_sme2_header_valid(
        const kevin_m4_q4k_sme2_header * header, const ggml_tensor * tensor) {
    return header != nullptr && header->magic == KEVIN_M4_Q4K_SME2_MAGIC &&
           header->version == KEVIN_M4_Q4K_SME2_VERSION &&
           header->n == static_cast<uint64_t>(tensor->ne[1]) &&
           header->k == static_cast<uint64_t>(tensor->ne[0]);
}

static block_q4_Kx8 kevin_m4_q4k_make_x8(const block_q4_K * input) {
    block_q4_Kx8 output{};
    for (int row = 0; row < 8; ++row) {
        output.d[row] = input[row].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        output.dmin[row] = input[row].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }
    for (int i = 0; i < QK_K * 4 / 8; ++i) {
        const int source_row = i % 8;
        const int source_offset = (i / 8) * 8;
        memcpy(output.qs + i * 8, input[source_row].qs + source_offset, 8);
    }

    uint8_t scales[8];
    uint8_t minimums[8];
    for (int group = 0; group < 4; ++group) {
        for (int row = 0; row < 8; ++row) {
            scales[row] = input[row].scales[group] & 63;
            minimums[row] = input[row].scales[group + 4] & 63;
        }
        uint8_t * destination = output.scales + group * 12;
        for (int row = 0; row < 4; ++row) {
            destination[row] = (scales[row] & 63) | ((scales[row + 4] & 48) << 2);
            destination[row + 4] = (minimums[row] & 63) | ((minimums[row + 4] & 48) << 2);
            destination[row + 8] = (scales[row + 4] & 15) | ((minimums[row + 4] & 15) << 4);
        }
    }
    for (int group = 0; group < 4; ++group) {
        for (int row = 0; row < 8; ++row) {
            scales[row] = ((input[row].scales[group] & 192) >> 2) |
                          (input[row].scales[group + 8] & 15);
            minimums[row] = ((input[row].scales[group + 4] & 192) >> 2) |
                            ((input[row].scales[group + 8] & 240) >> 4);
        }
        uint8_t * destination = output.scales + 48 + group * 12;
        for (int row = 0; row < 4; ++row) {
            destination[row] = (scales[row] & 63) | ((scales[row + 4] & 48) << 2);
            destination[row + 4] = (minimums[row] & 63) | ((minimums[row + 4] & 48) << 2);
            destination[row + 8] = (scales[row + 4] & 15) | ((minimums[row + 4] & 15) << 4);
        }
    }
    return output;
}

static void kevin_m4_q4k_repack_x8(
        const void * data, size_t n, size_t k, block_q4_Kx8 * destination) {
    const block_q4_K * source = static_cast<const block_q4_K *>(data);
    const size_t blocks = k / QK_K;
    block_q4_K rows[8];
    for (size_t row = 0; row < n; row += 8) {
        for (size_t block = 0; block < blocks; ++block) {
            for (int lane = 0; lane < 8; ++lane) {
                rows[lane] = source[(row + lane) * blocks + block];
            }
            *destination++ = kevin_m4_q4k_make_x8(rows);
        }
    }
}

static int kevin_m4_q4k_sme2_repack(
        ggml_tensor * tensor, const void * data, size_t data_size) {
    if (!kevin_m4_q4k_sme2_enabled() || !kevin_m4_q4k_sme2_tensor_eligible(tensor)) {
        return -1;
    }
    const size_t n = tensor->ne[1];
    const size_t k = tensor->ne[0];
    if (k % QK_K != 0 || data_size != ggml_nbytes(tensor)) {
        return -1;
    }

    memcpy(tensor->data, data, data_size);
    uint8_t * base = static_cast<uint8_t *>(tensor->data);
    kevin_m4_q4k_sme2_header * header = kevin_m4_q4k_sme2_header_for(tensor);
    const size_t header_offset = reinterpret_cast<uint8_t *>(header) - base;
    const size_t packed_offset =
        kevin_m4_q4k_sme2_align_up(header_offset + sizeof(*header));
    const size_t packed_size = kevin_m4_q4k_sme2_packed_size(n, k);
    const size_t correction_offset =
        kevin_m4_q4k_sme2_align_up(packed_offset + packed_size);
    const size_t groups = k / KEVIN_M4_Q4K_SME2_BL;
    const size_t correction_size = n * groups * sizeof(float);
    const size_t fallback_offset =
        kevin_m4_q4k_sme2_align_up(correction_offset + correction_size);

    std::vector<uint8_t> source(n * groups * KEVIN_M4_Q4K_SME2_SOURCE_BLOCK);
    float * correction = reinterpret_cast<float *>(base + correction_offset);
    const size_t row_bytes = (k / QK_K) * sizeof(block_q4_K);
    const uint8_t * input = static_cast<const uint8_t *>(data);

    for (size_t row = 0; row < n; ++row) {
        const block_q4_K * row_blocks =
            reinterpret_cast<const block_q4_K *>(input + row * row_bytes);
        for (size_t block = 0; block < k / QK_K; ++block) {
            const block_q4_K & source_block = row_blocks[block];
            const float d = GGML_FP16_TO_FP32(
                source_block.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
            const float dmin = GGML_FP16_TO_FP32(
                source_block.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
            for (int group = 0; group < 8; ++group) {
                uint8_t scale = 0;
                uint8_t minimum = 0;
                kevin_m4_q4k_decode_scale_min(source_block.scales, group, scale, minimum);
                const size_t group_index = block * 8 + group;
                uint8_t * destination = source.data() +
                    (row * groups + group_index) * KEVIN_M4_Q4K_SME2_SOURCE_BLOCK;
                const ggml_half scale_f16 = GGML_FP32_TO_FP16(d * static_cast<float>(scale));
                memcpy(destination, &scale_f16, sizeof(scale_f16));
                for (int i = 0; i < 16; ++i) {
                    const uint8_t low = kevin_m4_q4k_value(source_block, group, i);
                    const uint8_t high = kevin_m4_q4k_value(source_block, group, i + 16);
                    destination[sizeof(scale_f16) + i] = low | (high << 4);
                }
                correction[row * groups + group_index] =
                    8.0f * GGML_FP16_TO_FP32(scale_f16) -
                    dmin * static_cast<float>(minimum);
            }
        }
    }

    const size_t nr =
        kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const kai_rhs_pack_qs4cxs1s0_param pack_params{1, 8};
    kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
        1, n, k, nr, kr, sr, KEVIN_M4_Q4K_SME2_BL, source.data(), nullptr,
        base + packed_offset, 0, &pack_params);
    kevin_m4_q4k_repack_x8(
        data, n, k, reinterpret_cast<block_q4_Kx8 *>(base + fallback_offset));

    *header = {
        KEVIN_M4_Q4K_SME2_MAGIC,
        KEVIN_M4_Q4K_SME2_VERSION,
        n,
        k,
        packed_offset,
        packed_size,
        correction_offset,
        correction_size,
        fallback_offset,
        data_size,
    };
    return 0;
}

static bool kevin_m4_q4k_sme2_supports_op(const ggml_tensor * op) {
    if (!kevin_m4_q4k_sme2_enabled() || !ggml_cpu_has_sme() ||
        op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr ||
        !kevin_m4_q4k_sme2_tensor_eligible(op->src[0]) || op->src[0]->buffer == nullptr ||
        ggml_n_dims(op->src[0]) != 2 ||
        op->src[0]->buffer->buft != ggml_backend_cpu_kleidiai_buffer_type()) {
        return false;
    }
    return true;
}

static bool kevin_m4_q4k_sme2_is_decode_op(const ggml_tensor * op) {
    return kevin_m4_q4k_sme2_supports_op(op) &&
           op->src[1]->type == GGML_TYPE_F32 && op->src[1]->ne[1] == 1 &&
           op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1;
}

static bool kevin_m4_q4k_sme2_work_size(
        const ggml_tensor * op, size_t & size) {
    if (!kevin_m4_q4k_sme2_is_decode_op(op)) {
        return false;
    }
    const size_t k = op->src[0]->ne[0];
    const size_t mr =
        kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(
        1, k, KEVIN_M4_Q4K_SME2_BL, mr, kr, sr);
    const size_t sums_offset = kevin_m4_q4k_sme2_align_up(lhs_size);
    const size_t correction_values_offset = kevin_m4_q4k_sme2_align_up(
        sums_offset + (k / KEVIN_M4_Q4K_SME2_BL) * sizeof(float));
    const size_t sme_workspace = correction_values_offset +
        op->src[0]->ne[1] * sizeof(float);
    const size_t q8k_size = (k / QK_K) * sizeof(block_q8_K);
    const size_t q8k_copies = kevin_m4_q4k_sme2_shared_q8_enabled()
        ? 1
        : KEVIN_M4_Q4K_SME2_MAX_THREADS;
    size = kevin_m4_q4k_sme2_align_up(sme_workspace) +
           q8k_copies * q8k_size;
    return true;
}

static void kevin_m4_q4k_sme2_block_sums(
        const uint8_t * packed_lhs, size_t groups, float * sums) {
    const int8_t * values = reinterpret_cast<const int8_t *>(packed_lhs);
    const uint16_t * scales = reinterpret_cast<const uint16_t *>(
        packed_lhs + groups * KEVIN_M4_Q4K_SME2_BL);
    for (size_t group = 0; group < groups; ++group) {
        const int8_t * block = values + group * KEVIN_M4_Q4K_SME2_BL;
        const int sum = vaddlvq_s8(vld1q_s8(block)) + vaddlvq_s8(vld1q_s8(block + 16));
        sums[group] = static_cast<float>(sum) * GGML_FP16_TO_FP32(scales[group]);
    }
}

static float kevin_m4_q4k_sme2_correction_value(
        const float * coefficients, const float * sums, size_t row, size_t groups) {
    const float * coeff = coefficients + row * groups;
    float32x4_t total4 = vdupq_n_f32(0.0f);
    size_t group = 0;
    for (; group + 4 <= groups; group += 4) {
        total4 = vfmaq_f32(total4, vld1q_f32(coeff + group), vld1q_f32(sums + group));
    }
    float total = vaddvq_f32(total4);
    for (; group < groups; ++group) {
        total += coeff[group] * sums[group];
    }
    return total;
}

static void kevin_m4_q4k_sme2_correction_rows(
        const float * coefficients, const float * sums, size_t row_begin,
        size_t row_end, size_t groups, float * values, bool accumulate) {
    size_t row = row_begin;
    for (; row + 4 <= row_end; row += 4) {
        const float * coeff0 = coefficients + row * groups;
        const float * coeff1 = coeff0 + groups;
        const float * coeff2 = coeff1 + groups;
        const float * coeff3 = coeff2 + groups;
        float32x4_t total0 = vdupq_n_f32(0.0f);
        float32x4_t total1 = vdupq_n_f32(0.0f);
        float32x4_t total2 = vdupq_n_f32(0.0f);
        float32x4_t total3 = vdupq_n_f32(0.0f);
        size_t group = 0;
        for (; group + 4 <= groups; group += 4) {
            const float32x4_t sum4 = vld1q_f32(sums + group);
            total0 = vfmaq_f32(total0, vld1q_f32(coeff0 + group), sum4);
            total1 = vfmaq_f32(total1, vld1q_f32(coeff1 + group), sum4);
            total2 = vfmaq_f32(total2, vld1q_f32(coeff2 + group), sum4);
            total3 = vfmaq_f32(total3, vld1q_f32(coeff3 + group), sum4);
        }
        float corrected[4] = {
            vaddvq_f32(total0),
            vaddvq_f32(total1),
            vaddvq_f32(total2),
            vaddvq_f32(total3),
        };
        for (; group < groups; ++group) {
            corrected[0] += coeff0[group] * sums[group];
            corrected[1] += coeff1[group] * sums[group];
            corrected[2] += coeff2[group] * sums[group];
            corrected[3] += coeff3[group] * sums[group];
        }
        for (size_t lane = 0; lane < 4; ++lane) {
            if (accumulate) {
                values[row + lane] += corrected[lane];
            } else {
                values[row + lane] = corrected[lane];
            }
        }
    }
    for (; row < row_end; ++row) {
        const float corrected = kevin_m4_q4k_sme2_correction_value(
            coefficients, sums, row, groups);
        if (accumulate) {
            values[row] += corrected;
        } else {
            values[row] = corrected;
        }
    }
}

static void kevin_m4_q4k_sme2_quantize_q8k(
        const float * input, size_t k, block_q8_K * output) {
    for (size_t block = 0; block < k / QK_K; ++block) {
        const float * source = input + block * QK_K;
        block_q8_K & target = output[block];
        float32x4_t max4 = vdupq_n_f32(0.0f);
        for (int i = 0; i < QK_K; i += 16) {
            max4 = vmaxq_f32(max4, vabsq_f32(vld1q_f32(source + i)));
            max4 = vmaxq_f32(max4, vabsq_f32(vld1q_f32(source + i + 4)));
            max4 = vmaxq_f32(max4, vabsq_f32(vld1q_f32(source + i + 8)));
            max4 = vmaxq_f32(max4, vabsq_f32(vld1q_f32(source + i + 12)));
        }
        const float max_abs = vmaxvq_f32(max4);
        if (max_abs == 0.0f) {
            memset(&target, 0, sizeof(target));
            continue;
        }

        // Preserve the reference Q8_K sign rule: the first value with the
        // maximum magnitude determines the direction of the scale.
        float max_value = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            if (std::fabs(source[i]) == max_abs) {
                max_value = source[i];
                break;
            }
        }
        const float inverse_scale = -127.0f / max_value;
        const float32x4_t inverse4 = vdupq_n_f32(inverse_scale);
        const int32x4_t maximum4 = vdupq_n_s32(127);
        for (int i = 0; i < QK_K; i += 16) {
            const int32x4_t q0 = vminq_s32(
                maximum4, vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + i), inverse4)));
            const int32x4_t q1 = vminq_s32(
                maximum4, vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + i + 4), inverse4)));
            const int32x4_t q2 = vminq_s32(
                maximum4, vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + i + 8), inverse4)));
            const int32x4_t q3 = vminq_s32(
                maximum4, vcvtnq_s32_f32(vmulq_f32(vld1q_f32(source + i + 12), inverse4)));
            const int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            const int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            const int8x16_t quantized = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
            vst1q_s8(target.qs + i, quantized);
            target.bsums[i / 16] = vaddlvq_s8(quantized);
        }
        target.d = 1.0f / inverse_scale;
    }
}

static bool kevin_m4_q4k_sme2_compute(
        ggml_compute_params * params, ggml_tensor * dst) {
    if (!kevin_m4_q4k_sme2_is_decode_op(dst)) {
        return false;
    }
    const ggml_tensor * weights = dst->src[0];
    const kevin_m4_q4k_sme2_header * header = kevin_m4_q4k_sme2_header_for(weights);
    if (!kevin_m4_q4k_sme2_header_valid(header, weights)) {
        return false;
    }
    const ggml_tensor * activation = dst->src[1];
    const size_t n = weights->ne[1];
    const size_t k = weights->ne[0];
    const size_t groups = k / KEVIN_M4_Q4K_SME2_BL;
    const uint8_t * base = static_cast<const uint8_t *>(weights->data);
    const size_t mr =
        kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t kr =
        kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t sr =
        kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot();
    const size_t lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(
        1, k, KEVIN_M4_Q4K_SME2_BL, mr, kr, sr);
    uint8_t * packed_lhs = static_cast<uint8_t *>(params->wdata);
    const size_t sums_offset = kevin_m4_q4k_sme2_align_up(lhs_size);
    float * sums = reinterpret_cast<float *>(packed_lhs + sums_offset);

    const int nth = std::max(params->nth, 1);
    const int ith = params->ith;
    const size_t requested_sme_rows =
        (n * static_cast<size_t>(kevin_m4_q4k_sme2_share_percent()) + 99) / 100;
    const size_t sme_rows = nth == 1
        ? n
        : std::min(n, (requested_sme_rows + 7) & ~static_cast<size_t>(7));
    const size_t correction_values_offset = kevin_m4_q4k_sme2_align_up(
        sums_offset + groups * sizeof(float));
    float * correction_values = reinterpret_cast<float *>(
        packed_lhs + correction_values_offset);
    const size_t sme_workspace = kevin_m4_q4k_sme2_align_up(
        correction_values_offset + n * sizeof(float));
    const size_t q8k_size = (k / QK_K) * sizeof(block_q8_K);
    const bool shared_q8 = nth > 1 && sme_rows < n &&
        kevin_m4_q4k_sme2_shared_q8_enabled();
    const bool parallel_correction = nth > 1 && sme_rows < n &&
        kevin_m4_q4k_sme2_parallel_correction_enabled();
    block_q8_K * shared_q8k = reinterpret_cast<block_q8_K *>(
        static_cast<uint8_t *>(params->wdata) + sme_workspace);
    float * output = static_cast<float *>(dst->data);

    if (ith == 0) {
        kai_run_lhs_quant_pack_qsi8d32p_f32_neon(
            1, k, KEVIN_M4_Q4K_SME2_BL, mr, kr, sr, 0,
            static_cast<const float *>(activation->data), activation->nb[1], packed_lhs);
        kevin_m4_q4k_sme2_block_sums(packed_lhs, groups, sums);
    } else if (shared_q8 && ith == 1) {
        kevin_m4_q4k_sme2_quantize_q8k(
            static_cast<const float *>(activation->data), k, shared_q8k);
    }

    if (shared_q8 || parallel_correction) {
        ggml_barrier(params->threadpool);
    }

    if (ith == 0) {
        kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
            1, sme_rows, k, KEVIN_M4_Q4K_SME2_BL, packed_lhs, base + header->packed_offset,
            output, dst->nb[1], sizeof(float), -FLT_MAX, FLT_MAX);
        if (!parallel_correction) {
            const float * coefficients = reinterpret_cast<const float *>(
                base + header->correction_offset);
            kevin_m4_q4k_sme2_correction_rows(
                coefficients, sums, 0, sme_rows, groups, output, true);
        }
        kevin_m4_q4k_sme2_trace_once();
    } else if (ith < nth && sme_rows < n) {
        const size_t fallback_threads = static_cast<size_t>(nth - 1);
        const size_t fallback_index = static_cast<size_t>(ith - 1);
        const size_t correction_rows_per_thread =
            (sme_rows + fallback_threads - 1) / fallback_threads;
        const size_t correction_begin = std::min(
            sme_rows, fallback_index * correction_rows_per_thread);
        const size_t correction_end = std::min(
            sme_rows, correction_begin + correction_rows_per_thread);
        const float * coefficients = reinterpret_cast<const float *>(
            base + header->correction_offset);
        kevin_m4_q4k_sme2_correction_rows(
            coefficients, sums, correction_begin, correction_end, groups,
            correction_values, false);
        const size_t fallback_groups = (n - sme_rows) / 8;
        const size_t groups_per_thread =
            (fallback_groups + fallback_threads - 1) / fallback_threads;
        const size_t row_begin = std::min(
            n, sme_rows + fallback_index * groups_per_thread * 8);
        const size_t row_end = std::min(n, row_begin + groups_per_thread * 8);
        block_q8_K * q8k = shared_q8k;
        if (!shared_q8) {
            q8k = reinterpret_cast<block_q8_K *>(
                static_cast<uint8_t *>(params->wdata) +
                sme_workspace + fallback_index * q8k_size);
            kevin_m4_q4k_sme2_quantize_q8k(
                static_cast<const float *>(activation->data), k, q8k);
        }
        if (row_begin < row_end) {
            const size_t blocks = k / QK_K;
            const block_q4_Kx8 * fallback = reinterpret_cast<const block_q4_Kx8 *>(
                base + header->fallback_offset) + (row_begin / 8) * blocks;
            ggml_gemv_q4_K_8x8_q8_K(
                static_cast<int>(k), output + row_begin, n, fallback, q8k,
                1, static_cast<int>(row_end - row_begin));
        }
    }
    if (parallel_correction) {
        ggml_barrier(params->threadpool);
        const size_t rows_per_thread =
            (sme_rows + static_cast<size_t>(nth) - 1) / static_cast<size_t>(nth);
        const size_t row_begin = std::min(
            sme_rows, static_cast<size_t>(ith) * rows_per_thread);
        const size_t row_end = std::min(sme_rows, row_begin + rows_per_thread);
        for (size_t row = row_begin; row < row_end; ++row) {
            output[row] += correction_values[row];
        }
    }
    return true;
}

#endif

#ifndef KEVIN_M4_Q4K_H
#define KEVIN_M4_Q4K_H

#include <arm_neon.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int kevin_m4_q4k_enabled_state = 0;
static int kevin_m4_q4k_trace_state = 0;

__attribute__((constructor)) static void kevin_m4_q4k_initialize(void) {
    const char * enabled = getenv("GGML_M4_Q4K_CUSTOM");
    const char * trace = getenv("GGML_M4_Q4K_TRACE");
    kevin_m4_q4k_enabled_state =
            enabled != NULL && enabled[0] == '1' && enabled[1] == '\0';
    kevin_m4_q4k_trace_state =
            trace != NULL && trace[0] == '1' && trace[1] == '\0';
}

static int kevin_m4_q4k_enabled(void) {
    return kevin_m4_q4k_enabled_state;
}

static void kevin_m4_q4k_trace_once(void) {
    static atomic_flag emitted = ATOMIC_FLAG_INIT;
    if (kevin_m4_q4k_trace_state &&
            !atomic_flag_test_and_set_explicit(&emitted, memory_order_relaxed)) {
        fprintf(stderr, "kevin_m4_q4k: custom Q4_K x Q8_K decode kernel hit\n");
    }
}

static void kevin_m4_vec_dot_q4_K_q8_K(
        int n, float * s, const block_q4_K * x, const block_q8_K * y) {
    const int nb = n / QK_K;
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    const uint32_t mask1 = 0x3f3f3f3f;
    const uint32_t mask2 = 0x0f0f0f0f;
    const uint32_t mask3 = 0x03030303;
    float total = 0.0f;

    kevin_m4_q4k_trace_once();
    for (int block = 0; block < nb; ++block) {
        uint32_t decoded[4] = {0, 0, 0, 0};
        memcpy(decoded, x[block].scales, K_SCALE_SIZE);
        const uint32_t packed_mins = decoded[1] & mask1;
        decoded[3] = ((decoded[2] >> 4) & mask2) |
                     (((decoded[1] >> 6) & mask3) << 4);
        decoded[1] = (decoded[2] & mask2) |
                     (((decoded[0] >> 6) & mask3) << 4);
        decoded[2] = packed_mins;
        decoded[0] &= mask1;
        const uint8_t * scales = (const uint8_t *) &decoded[0];
        const uint8_t * mins = (const uint8_t *) &decoded[2];

        const int16x8_t q8_sums = vpaddq_s16(
                vld1q_s16(y[block].bsums), vld1q_s16(y[block].bsums + 8));
        const int16x8_t min_values = vreinterpretq_s16_u16(
                vmovl_u8(vld1_u8(mins)));
        const int32x4_t min_products = vaddq_s32(
                vmull_s16(vget_low_s16(q8_sums), vget_low_s16(min_values)),
                vmull_s16(vget_high_s16(q8_sums), vget_high_s16(min_values)));
        const int weighted_min = vaddvq_s32(min_products);
        int weighted_low = 0;
        int weighted_high = 0;

        for (int chunk = 0; chunk < QK_K / 64; ++chunk) {
            const uint8x16x2_t packed = vld1q_u8_x2(x[block].qs + chunk * 32);
            const int8x16_t low0 = vreinterpretq_s8_u8(
                    vandq_u8(packed.val[0], mask));
            const int8x16_t low1 = vreinterpretq_s8_u8(
                    vandq_u8(packed.val[1], mask));
            const int8x16_t high0 = vreinterpretq_s8_u8(
                    vshrq_n_u8(packed.val[0], 4));
            const int8x16_t high1 = vreinterpretq_s8_u8(
                    vshrq_n_u8(packed.val[1], 4));
            const int8_t * q8 = y[block].qs + chunk * 64;
            int32x4_t low_dot = vdotq_s32(
                    vdupq_n_s32(0), low0, vld1q_s8(q8));
            low_dot = vdotq_s32(low_dot, low1, vld1q_s8(q8 + 16));
            int32x4_t high_dot = vdotq_s32(
                    vdupq_n_s32(0), high0, vld1q_s8(q8 + 32));
            high_dot = vdotq_s32(high_dot, high1, vld1q_s8(q8 + 48));
            weighted_low += vaddvq_s32(low_dot) * scales[chunk * 2];
            weighted_high += vaddvq_s32(high_dot) * scales[chunk * 2 + 1];
        }
        total += GGML_CPU_FP16_TO_FP32(x[block].d) * y[block].d *
                         (weighted_low + weighted_high) -
                 GGML_CPU_FP16_TO_FP32(x[block].dmin) * y[block].d * weighted_min;
    }
    *s = total;
}

#endif

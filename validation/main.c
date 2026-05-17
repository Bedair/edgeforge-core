/**
 * EdgeForge Validation Harness
 * ============================
 * Minimal C program that includes all EdgeForge generated files and
 * exercises the full API with a dummy input.
 *
 * Purpose:
 *   Proves the generated files are syntactically correct C and that
 *   the API is usable before flashing to real hardware.
 *
 * Compile (host, for quick syntax check):
 *   gcc -std=c11 -I edgeforge_output/ validation/main.c -o /dev/null -fsyntax-only
 *
 * Compile (target, for binary size estimate):
 *   arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
 *     -mthumb -std=c11 -Os -I edgeforge_output/
 *     -c validation/main.c -o validation/main.o
 *
 * On real hardware, link against:
 *   - TFLite Micro static library
 *   - Your BSP / CMSIS / RTOS libraries
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* EdgeForge generated headers */
#include "model.h"
#include "memory_config.h"
#include "inference_runner.h"

/* ── Static allocations (would live in your firmware's .bss section) ──────── */

/* Model handle -- contains the arena, interpreter pointer, and init flag */
static edgeforge_model_t g_model;

/* Input buffer -- fill with your preprocessed sensor data before inference */
static edgeforge_input_t  g_input[EDGEFORGE_INPUT_SIZE  / sizeof(edgeforge_input_t)];

/* Output buffer -- read classification scores after inference */
static edgeforge_output_t g_output[EDGEFORGE_OUTPUT_SIZE / sizeof(edgeforge_output_t)];

/* ── Validation helpers ───────────────────────────────────────────────────── */

static void print_status(const char *step, edgeforge_status_t status) {
#ifdef EDGEFORGE_HOST_VALIDATION
    if (status == EDGEFORGE_OK) {
        printf("[OK]   %s\n", step);
    } else {
        printf("[FAIL] %s (error %d)\n", step, (int)status);
    }
#else
    /* On device: send over UART */
    (void)step;
    (void)status;
#endif
}

/* ── Dummy input generation ──────────────────────────────────────────────────
 * On real hardware, replace this with actual sensor data:
 *   - Audio: MFCC features from microphone
 *   - Vision: normalised camera frame
 *   - IMU: accelerometer window
 */
static void fill_dummy_input(void) {
    /* Fill with a simple ramp pattern for validation */
    size_t n = EDGEFORGE_INPUT_SIZE / sizeof(edgeforge_input_t);
    for (size_t i = 0; i < n; i++) {
#if EDGEFORGE_IS_QUANTIZED
        /* INT8 input: values in [-128, 127] */
        ((int8_t *)g_input)[i] = (int8_t)(i % 256 - 128);
#else
        /* float32 input: values in [-1.0, 1.0] */
        ((float *)g_input)[i] = (float)(i % 100) / 50.0f - 1.0f;
#endif
    }
}

/* ── Output reading ──────────────────────────────────────────────────────────
 * Returns the index of the highest-scoring class.
 */
static int argmax(void) {
    size_t n = EDGEFORGE_OUTPUT_SIZE / sizeof(edgeforge_output_t);
    int    best_idx   = 0;
    float  best_score = -1e9f;

    for (size_t i = 0; i < n; i++) {
#if EDGEFORGE_IS_QUANTIZED
        float score = ((int8_t *)g_output)[i] * EDGEFORGE_OUTPUT_SCALE
                      + EDGEFORGE_OUTPUT_ZERO_POINT;
#else
        float score = ((float *)g_output)[i];
#endif
        if (score > best_score) {
            best_score = score;
            best_idx   = (int)i;
        }
    }
    return best_idx;
}

/* ── Main validation sequence ────────────────────────────────────────────────
 * Call from your firmware's main() or from a FreeRTOS/Zephyr task.
 */
int edgeforge_validate(void) {
    edgeforge_status_t status;

    /* Step 1: Initialise model */
    status = edgeforge_init(&g_model);
    print_status("edgeforge_init", status);
    if (status != EDGEFORGE_OK) return -1;

    /* Step 2: Fill dummy input */
    fill_dummy_input();

    /* Step 3: Run inference */
    status = edgeforge_infer(&g_model, g_input, g_output);
    print_status("edgeforge_infer", status);
    if (status != EDGEFORGE_OK) return -2;

    /* Step 4: Read result */
    int class_idx = argmax();
    (void)class_idx;  /* Use on device: send over UART */

#ifdef EDGEFORGE_HOST_VALIDATION
    printf("Result: class %d\n", class_idx);
    printf("Arena:  %u bytes used of %u available\n",
           EDGEFORGE_ARENA_SIZE,
           (unsigned)(EDGEFORGE_TARGET_RAM_KB * 1024U));
#endif

    /* Step 5: Optional deinit */
    edgeforge_deinit(&g_model);
    print_status("edgeforge_deinit", EDGEFORGE_OK);

    return 0;
}

/* ── Entry point for host-side syntax validation ─────────────────────────── */
#ifdef EDGEFORGE_HOST_VALIDATION
int main(void) {
    printf("EdgeForge Validation Harness\n");
    printf("Model:  " EDGEFORGE_MODEL_NAME "\n");
    printf("Arena:  %u bytes (%.1f KB)\n",
           EDGEFORGE_ARENA_SIZE,
           (float)EDGEFORGE_ARENA_SIZE / 1024.0f);
    printf("Input:  %u bytes\n", EDGEFORGE_INPUT_SIZE);
    printf("Output: %u bytes\n", EDGEFORGE_OUTPUT_SIZE);
    printf("Quantised: %s\n\n", EDGEFORGE_IS_QUANTIZED ? "yes" : "no");
    return edgeforge_validate();
}
#endif /* EDGEFORGE_HOST_VALIDATION */

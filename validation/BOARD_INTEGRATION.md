# EdgeForge Board Integration Guide
## Phase 4 -- Validating generated code on real hardware

---

## Prerequisites

### 1. Generate the output files

```bash
# Generate and optimize the model
python generate_test_model.py
edgeforge optimize test_model.onnx --mcu=stm32f407
edgeforge compile  test_model_opt.onnx --mcu=stm32f407 --rtos=freertos

# Validate the output compiles
edgeforge benchmark edgeforge_output/ --mcu=stm32f407
```

### 2. Get TFLite Micro

TFLite Micro is the one dependency EdgeForge does not generate.
Download the pre-built static library for Cortex-M4:

```bash
# Option A: Use the TFLite Micro CMake build
git clone https://github.com/tensorflow/tflite-micro.git
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn \
  microlite

# Option B: Use the Arduino TensorFlowLite library (easiest for prototyping)
# In Arduino IDE: Tools -> Manage Libraries -> search "Arduino_TensorFlowLite"
```

---

## Board A: STM32F407 Discovery / Nucleo

### Toolchain
- STM32CubeIDE or `arm-none-eabi-gcc` + Makefile
- ST-Link programmer (built into Discovery board)
- Baud: 115200, UART2 (PA2/PA3 on Discovery)

### Project structure

```
your_stm32_project/
+-- Core/
|   +-- Src/
|   |   +-- main.c              (your firmware main)
|   |   +-- edgeforge_validate.c (copy validation/main.c here)
|   +-- Inc/
+-- edgeforge_output/           (copy here from CLI output)
|   +-- model.h
|   +-- model.c
|   +-- memory_config.h
|   +-- inference_runner.h
|   +-- inference_runner.c
|   +-- rtos_glue.c
|   +-- CMakeLists.txt
+-- Middlewares/
|   +-- tflite-micro/           (TFLite Micro static lib)
+-- CMakeLists.txt
```

### CMakeLists.txt additions

```cmake
# Add EdgeForge
add_subdirectory(edgeforge_output)
target_link_libraries(${PROJECT_NAME} edgeforge_test_model_opt)

# Add TFLite Micro
target_include_directories(${PROJECT_NAME} PRIVATE
    Middlewares/tflite-micro
)
target_link_libraries(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/Middlewares/tflite-micro/libtensorflow-microlite.a
)
```

### main.c integration

```c
#include "inference_runner.h"

/* Declare the validation function from validation/main.c */
extern int edgeforge_validate(void);

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_USART2_UART_Init();   /* 115200 baud */

    /* Run EdgeForge validation */
    int result = edgeforge_validate();

    /* Blink LED based on result */
    while (1) {
        HAL_GPIO_TogglePin(LD4_GPIO_Port, LD4_Pin);
        HAL_Delay(result == 0 ? 500 : 100);  /* slow=OK, fast=FAIL */
    }
}
```

### Expected UART output

```
EdgeForge Validation Harness
Model:  test_model_opt
Arena:  13928 bytes (13.6 KB)
Input:  1960 bytes
Output: 40 bytes
Quantised: yes

[OK]   edgeforge_init
[OK]   edgeforge_infer
Result: class 3
Arena:  13928 bytes used of 196608 available
[OK]   edgeforge_deinit
```

### Measuring latency (DWT cycle counter)

```c
#include "inference_runner.h"

void measure_inference_latency(void) {
    /* Enable DWT cycle counter */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

    static edgeforge_model_t model;
    static edgeforge_input_t input[EDGEFORGE_INPUT_SIZE / sizeof(edgeforge_input_t)];
    static edgeforge_output_t output[EDGEFORGE_OUTPUT_SIZE / sizeof(edgeforge_output_t)];

    edgeforge_init(&model);

    uint32_t t_start = DWT->CYCCNT;
    edgeforge_infer(&model, input, output);
    uint32_t t_end   = DWT->CYCCNT;

    uint32_t cycles  = t_end - t_start;
    uint32_t ms      = cycles / (SystemCoreClock / 1000);

    char buf[64];
    sprintf(buf, "Inference: %lu ms (%lu cycles)\r\n", ms, cycles);
    HAL_UART_Transmit(&huart2, (uint8_t*)buf, strlen(buf), HAL_MAX_DELAY);
}
```

---

## Board B: Infineon PSoC6

### Toolchain
- ModusToolbox 3.x or `arm-none-eabi-gcc` + Makefile
- KitProg3 programmer (built into CY8CKIT-062-WiFi-BT)

### Key differences from STM32

```c
/* PSoC6 UART init (ModusToolbox generated) */
cy_rslt_t result = cy_retarget_io_init(CYBSP_DEBUG_UART_TX,
                                        CYBSP_DEBUG_UART_RX,
                                        CY_RETARGET_IO_BAUDRATE);

/* printf goes to UART automatically after cy_retarget_io_init */
edgeforge_validate();  /* uses printf internally */
```

### Generate for PSoC6

```bash
edgeforge compile test_model_opt.onnx --mcu=psoc6 --rtos=freertos
edgeforge benchmark edgeforge_output/ --mcu=psoc6
```

---

## Board C: Nordic nRF52840 (Arduino Nano 33 BLE Sense)

### Toolchain
- Zephyr RTOS + west build system
- nRF52840-DK or Arduino Nano 33 BLE Sense
- JLink programmer or UF2 bootloader

### Generate for nRF52840

```bash
edgeforge compile test_model_opt.onnx --mcu=nrf52840 --rtos=zephyr
edgeforge benchmark edgeforge_output/ --mcu=nrf52840
```

### Zephyr CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(edgeforge_demo)

target_sources(app PRIVATE
    src/main.c
    edgeforge_output/model.c
    edgeforge_output/inference_runner.c
    edgeforge_output/rtos_glue.c
)
target_include_directories(app PRIVATE edgeforge_output/)
```

### Zephyr main.c

```c
#include <zephyr/kernel.h>
#include "inference_runner.h"

extern int edgeforge_validate(void);

int main(void) {
    printk("EdgeForge on nRF52840\n");
    edgeforge_rtos_start();
    edgeforge_validate();
    return 0;
}
```

---

## Benchmark results table

Fill this in as you validate each board:

| Board | MCU | Inference (ms) | RAM used | Flash used | Status |
|---|---|---|---|---|---|
| STM32F407 Discovery | Cortex-M4F @168MHz | ? | 13.6 KB | ? KB | Pending |
| Infineon PSoC6 | Cortex-M4F @150MHz | ? | 13.6 KB | ? KB | Pending |
| Arduino Nano 33 BLE | Cortex-M4F @64MHz | ? | 13.6 KB | ? KB | Pending |

---

*Generated by EdgeForge Phase 4 validation guide.*

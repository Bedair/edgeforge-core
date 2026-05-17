# EdgeForge -- STM32F407 Integration Guide
# TFLite Micro from scratch + STM32CubeIDE

---

## Overview

This guide sets up:
  1. STM32CubeIDE project for STM32F407 Discovery
  2. TFLite Micro static library (pre-built, no CMake needed)
  3. EdgeForge generated files
  4. UART output to verify inference works

Total time: ~2 hours first time, ~10 minutes for subsequent models.

---

## Part 1 -- Get TFLite Micro (pre-built, easiest path)

Building TFLite Micro from source inside STM32CubeIDE is painful.
Use the pre-built static library from the EloquentTinyML project instead --
it is the same TFLite Micro, just pre-compiled for Cortex-M4.

### Download

Go to: https://github.com/eloquentarduino/tflite-micro-compiled

Download the Cortex-M4F release:
  - tflite-micro-cortex-m4f.zip (contains libtensorflow-microlite.a + headers)

Extract to: your_project/Middlewares/tflite-micro/

Structure after extraction:
  Middlewares/
  +-- tflite-micro/
      +-- include/
      |   +-- tensorflow/
      |       +-- lite/
      |           +-- micro/
      |               +-- micro_interpreter.h
      |               +-- micro_mutable_op_resolver.h
      |               +-- micro_error_reporter.h
      |               +-- all_ops_resolver.h
      |               +-- ...
      |           +-- schema/
      |               +-- schema_generated.h
      +-- lib/
          +-- cortex-m4f/
              +-- libtensorflow-microlite.a

### Alternative: Use STM32Cube.AI backend (no TFLite Micro needed)

If the above is too complex, STM32Cube.AI (X-CUBE-AI) generates its own
C runtime and works inside CubeIDE natively. In that case, EdgeForge's
model.c and memory_config.h are still useful, but inference_runner.c
should be replaced with the Cube.AI generated runner.

For this guide we use raw TFLite Micro.

---

## Part 2 -- Create STM32CubeIDE project

### Step 1: New project

  File -> New -> STM32 Project
  Board: STM32F407G-DISC1  (or your specific board)
  Language: C++  (TFLite Micro requires C++)
  Name: edgeforge_demo

### Step 2: Enable UART in CubeMX

  In the .ioc file:
  - Connectivity -> USART2 -> Mode: Asynchronous
  - Baud: 115200, Word Length: 8, Parity: None, Stop Bits: 1
  - PA2 = USART2_TX, PA3 = USART2_RX (Discovery board defaults)
  - Click: Generate Code

### Step 3: Increase stack and heap

  In STM32F407VGTX_FLASH.ld (or your linker script):

  _Min_Heap_Size  = 0x8000;   /* 32KB -- TFLite Micro needs heap */
  _Min_Stack_Size = 0x2000;   /* 8KB  */

### Step 4: Add FPU flags (CRITICAL for performance)

  Right-click project -> Properties
  -> C/C++ Build -> Settings
  -> MCU/MPU GCC Compiler -> General
  -> Floating-point unit: FPv4-SP-D16
  -> Floating-point ABI: Hard

  Same for MCU/MPU G++ Compiler.

---

## Part 3 -- Add TFLite Micro to the project

### Step 1: Add include path

  Project Properties -> C/C++ Build -> Settings
  -> MCU GCC Compiler -> Include Paths
  -> Add: ../Middlewares/tflite-micro/include

  Same for G++ Compiler.

### Step 2: Add static library

  Project Properties -> C/C++ Build -> Settings
  -> MCU G++ Linker -> Libraries
  -> Library search path (-L): ../Middlewares/tflite-micro/lib/cortex-m4f
  -> Libraries (-l): tensorflow-microlite

### Step 3: Add required linker flags

  MCU G++ Linker -> Miscellaneous -> Other flags:
  -Wl,--gc-sections -lm -lstdc++ -lsupc++

---

## Part 4 -- Add EdgeForge generated files

### Step 1: Generate the files

  In your EdgeForge environment:
    edgeforge compile test_model_opt.onnx --mcu=stm32f407 --rtos=freertos

### Step 2: Copy to project

  Copy the entire edgeforge_output/ folder into your project:
    your_project/
    +-- edgeforge_output/
        +-- model.h
        +-- model.c
        +-- memory_config.h
        +-- inference_runner.h
        +-- inference_runner.c
        +-- rtos_glue.c
        +-- CMakeLists.txt  (ignored by CubeIDE -- for reference only)
        +-- README.md

  In STM32CubeIDE: right-click project -> Refresh

### Step 3: Add include path for EdgeForge

  Project Properties -> C/C++ Build -> Settings
  -> MCU GCC Compiler -> Include Paths -> Add: ../edgeforge_output

  Same for G++ Compiler.

### Step 4: Add source files to build

  Right-click edgeforge_output/ -> Properties
  -> C/C++ Build -> Uncheck "Exclude resource from build"

  This tells CubeIDE to compile model.c, inference_runner.c, rtos_glue.c.

---

## Part 5 -- Write the test firmware

### Copy validation/main.c

Copy validation/main.c from the EdgeForge repo into your project:
  your_project/Core/Src/edgeforge_validate.c

No changes needed -- it includes all the right headers already.

### Edit Core/Src/main.c

Add these includes at the top of main.c (inside the USER CODE BEGIN Includes):

```c
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "inference_runner.h"
/* USER CODE END Includes */
```

Add UART printf redirect (inside USER CODE BEGIN 0):

```c
/* USER CODE BEGIN 0 */
int __io_putchar(int ch) {
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

extern int edgeforge_validate(void);
/* USER CODE END 0 */
```

Call validation in the main loop (inside USER CODE BEGIN 2):

```c
/* USER CODE BEGIN 2 */
printf("EdgeForge STM32F407 Demo\r\n");
printf("========================\r\n");

int result = edgeforge_validate();

if (result == 0) {
    printf("\r\nSUCCESS -- inference running correctly\r\n");
} else {
    printf("\r\nFAIL -- error code: %d\r\n", result);
}
/* USER CODE END 2 */
```

Add LED blink in the while(1) (inside USER CODE BEGIN 3):

```c
/* USER CODE BEGIN 3 */
HAL_GPIO_TogglePin(LD4_GPIO_Port, LD4_Pin);
HAL_Delay(result == 0 ? 1000 : 100);
/* USER CODE END 3 */
```

---

## Part 6 -- Build and flash

### Build

  Project -> Build Project  (or Ctrl+B)

  Expected: 0 errors.
  Common warnings (safe to ignore):
  - -Wunused-parameter in TFLite Micro headers
  - -Wmissing-field-initializers in generated code

### Flash

  Run -> Debug  (F11)
  This flashes via ST-Link and starts a debug session.

### Check UART output

  Open a serial terminal:
  - Windows: PuTTY or Tera Term
  - Port: COMx (check Device Manager)
  - Baud: 115200
  - No parity, 8 data bits, 1 stop bit

  Expected output:
    EdgeForge STM32F407 Demo
    ========================
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

    SUCCESS -- inference running correctly

---

## Part 7 -- Measure inference latency

Replace edgeforge_validate() call with this latency measurement:

```c
/* USER CODE BEGIN 2 */
#include "dwt_delay.h"  /* or use HAL_GetTick() */

static edgeforge_model_t model;
static edgeforge_input_t input[EDGEFORGE_INPUT_SIZE / sizeof(edgeforge_input_t)];
static edgeforge_output_t output[EDGEFORGE_OUTPUT_SIZE / sizeof(edgeforge_output_t)];

/* Enable DWT cycle counter */
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

edgeforge_init(&model);

/* Warm up (first inference is slower due to cache) */
edgeforge_infer(&model, input, output);

/* Measured inference */
uint32_t t0 = DWT->CYCCNT;
edgeforge_infer(&model, input, output);
uint32_t t1 = DWT->CYCCNT;

uint32_t cycles = t1 - t0;
uint32_t us     = cycles / (SystemCoreClock / 1000000);
uint32_t ms     = us / 1000;

printf("Inference latency: %lu ms (%lu us, %lu cycles)\r\n", ms, us, cycles);
printf("System clock:      %lu MHz\r\n", SystemCoreClock / 1000000);
/* USER CODE END 2 */
```

---

## Troubleshooting

### "undefined reference to tflite::"
  -> TFLite Micro library not linked. Check Part 3 Step 2.

### "HardFault on edgeforge_init"
  -> Arena too small or not aligned. Check EDGEFORGE_ARENA_SIZE in memory_config.h.
  -> Try increasing heap: _Min_Heap_Size = 0x10000 in linker script.

### "No UART output"
  -> Check __io_putchar redirect is in main.c.
  -> Check baud rate matches (115200).
  -> Check PA2/PA3 are configured as USART2 in .ioc file.

### "Build error: cannot open source file tensorflow/lite/..."
  -> Include path for TFLite Micro headers not added. Check Part 3 Step 1.

### "Inference returns wrong class every time"
  -> Normal -- dummy input produces arbitrary output.
  -> Replace fill_dummy_input() with real sensor data.

---

## Success criteria for Phase 4 M2

  [x] Project builds with 0 errors
  [x] UART shows "[OK] edgeforge_init"
  [x] UART shows "[OK] edgeforge_infer"
  [x] Inference latency measured and recorded
  [x] LED blinks at 1Hz (success indicator)

Record the latency result for the benchmark table in BOARD_INTEGRATION.md.

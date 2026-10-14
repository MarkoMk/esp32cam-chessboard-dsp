#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/sdmmc_host.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

/* DEFINE PINS */

#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0      5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

/* FLASH PIN */

#define STATUS_LED_GPIO GPIO_NUM_4

/* TEST CONFIG */

#define MOUNT_POINT "/sdcard"
#define WARMUP_FRAMES           5
#define TEST_FRAMES             30
#define SAVE_EVERY_N            5

#define IMG_WIDTH               160
#define IMG_HEIGHT              120
#define IMG_PIXELS              (IMG_WIDTH * IMG_HEIGHT)

#define EDGE_THRESHOLD          110

/* HOUGH CONFIG */

#define HOUGH_THETA_BINS        90
#define HOUGH_THETA_STEP_DEG    2
#define HOUGH_RHO_MAX           200
#define HOUGH_RHO_BINS          401
#define HOUGH_MIN_VOTES         40
#define HOUGH_MAX_LINES         20
#define HOUGH_SUPPRESS_THETA    4
#define HOUGH_SUPPRESS_RHO      8

/* HOUGH FAMILY CONFIG */

#define FAMILY_SCORE_RADIUS         4
#define FAMILY_ANGLE_WINDOW         6
#define FAMILY_MIN_SEPARATION_BINS  35
#define FAMILY_MAX_LINES            12

/* FAMILY TRACKING */

#define FAMILY_TRACK_MAX_JUMP_BINS  8
#define FAMILY_TRACK_MIN_LINES      2
#define FAMILY_TRACK_RESET_MISSES   100

/* ROI CONFIG */

#define ROI_BLOCK_SIZE          10
#define ROI_BLOCK_COLS          (IMG_WIDTH / ROI_BLOCK_SIZE)
#define ROI_BLOCK_ROWS          (IMG_HEIGHT / ROI_BLOCK_SIZE)
#define ROI_TOTAL_BLOCKS        (ROI_BLOCK_COLS * ROI_BLOCK_ROWS)

#define ROI_DENSE_EDGE_COUNT    60
#define ROI_MIN_BLOCKS          6
#define ROI_EXPAND_BLOCKS       1

/* BOARD CONFIG */

#define BOARD_MIN_FAMILY_SPAN       30
#define BOARD_CORNER_MARGIN         5
#define BOARD_MIN_AREA              1200.0f
#define BOARD_MIN_ROI_COVERAGE      60
#define BOARD_MIN_ROI_PRECISION     45

/* OLD HOUGH GRID CONFIG */

#define GRID_MIN_SPACING        6
#define GRID_MAX_SPACING        18
#define GRID_RHO_TOLERANCE      3
#define GRID_MIN_PEAK_VOTES     12
#define GRID_MIN_MATCHED_LINES  7

#define GRID_MAX_CANDIDATES     8
#define GRID_DUPLICATE_RHO      5

#define GRID_PROJECTION_HIST_BINS   256
#define GRID_PEAK_MIN_SEPARATION    7
#define GRID_ASSIGN_MAX_DISTANCE    5.0f

/* OUTER LINE CONFIG */

#define OUTER_LINE_MAX              32
#define OUTER_PAIR_MAX              12
#define OUTER_MIN_PEAK_VOTES        12
#define OUTER_RHO_NMS               3

#define PI_F 3.14159265358979323846f

/* HARRIS CONFIG */

#define HARRIS_WINDOW_RADIUS        1
#define HARRIS_K_DIVISOR            25
#define HARRIS_RESPONSE_THRESHOLD   35

#define CHESS_CORNER_NMS_RADIUS     2
#define CHESS_CORNER_MAX            256

/* NEW CHESS GRID CONFIG */

#define GRID_LINE_COUNT             9
#define GRID_MIN_POINTS_PER_LINE    2
#define GRID_MAX_LINE_ANGLE_ERROR_DEG 12.0f

#define GRID_ANGLE_BINS             90
#define GRID_PAIR_MIN_DIST          6
#define GRID_PAIR_MAX_DIST          13
#define GRID_AXIS_MIN_SEP_BINS      30

/* PERSPECTIVE CORRECTION */

#define BOARD_WARP_SIZE             96
#define BOARD_WARP_PIXELS           (BOARD_WARP_SIZE * BOARD_WARP_SIZE)

/* GLOBALS */

static const char *TAG = "DSP_BENCH";

static uint8_t *gray_buffer = NULL;
static uint8_t *blur_buffer = NULL;
static uint8_t *gradient_buffer = NULL;
static uint8_t *edge_buffer = NULL;
static uint8_t *board_warp_buffer = NULL;

/* BASIC POINT TYPE */

typedef struct {
    float x;
    float y;
} point_f_t;

/* HOUGH LINE */

typedef struct {
    int theta_index;
    int rho;
    uint8_t votes;
} hough_line_t;

/* HARRIS CORNER */

typedef struct {
    int x;
    int y;
    uint8_t score;
} chess_corner_t;

/* NEW GRID LINE */

typedef struct {
    float a;
    float b;
    float c;

    int valid;
    int point_count;
} grid_line_t;

/* NEW CHESS GRID */

typedef struct {
    grid_line_t rows[GRID_LINE_COUNT];
    grid_line_t cols[GRID_LINE_COUNT];

    point_f_t corners[4];

    int valid;
} chess_grid_t;

typedef struct {
    float a;
    float b;
    float c;

    float d;
    float e;
    float f;

    float g;
    float h;
} projective_map_t;

/* HOUGH OUTER-LINE PAIR */

typedef struct {
    hough_line_t first;
    hough_line_t last;

    int span;
    uint32_t score;
} line_pair_candidate_t;

/* OLD HOUGH GRID CANDIDATE */

typedef struct {
    hough_line_t first;
    hough_line_t last;

    int spacing;
    int matched;

    uint32_t score;
} grid_candidate_t;

/* OLD HOUGH BOARD DETECTION */

typedef struct {
    point_f_t corner[4];

    hough_line_t a_min;
    hough_line_t a_max;

    hough_line_t b_min;
    hough_line_t b_max;

    int valid;
} board_detection_t;

/* EDGE ROI */

typedef struct {
    int x0;
    int y0;

    int x1;
    int y1;

    int dense_blocks;
    int valid;
} edge_roi_t;

static uint8_t *hough_accumulator = NULL;
static uint8_t *line_buffer = NULL;

static int16_t hough_cos_q15[HOUGH_THETA_BINS];
static int16_t hough_sin_q15[HOUGH_THETA_BINS];

static chess_corner_t chess_corners[CHESS_CORNER_MAX];

static hough_line_t hough_lines[HOUGH_MAX_LINES];

static hough_line_t family_a_lines[FAMILY_MAX_LINES];
static hough_line_t family_b_lines[FAMILY_MAX_LINES];
static uint32_t theta_scores[HOUGH_THETA_BINS];

static int tracked_family_a_theta = -1;
static int tracked_family_b_theta = -1;
static int family_track_misses = 0;

static uint8_t roi_block_mask[ROI_TOTAL_BLOCKS];

static chess_grid_t detected_grid;

/* LED HELPERS */

static void led_init(void)
{
    gpio_reset_pin(STATUS_LED_GPIO);
    gpio_set_direction(STATUS_LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(STATUS_LED_GPIO, 0);
}

static void blink_once(int on_ms, int off_ms)
{
    gpio_set_level(STATUS_LED_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(on_ms));
    gpio_set_level(STATUS_LED_GPIO, 0);
    vTaskDelay(pdMS_TO_TICKS(off_ms));
}

static int warp_chessboard(
    const uint8_t *source,
    uint8_t *destination,
    const chess_grid_t *grid
);

/*
    ERROR CODES:
    3 flashes = SD initialization failure
    4 flashes = Benchmark/file failure
    5 flashes = Camera initialization failure
    6 flashes = DSP memory allocation failure
*/

static void blink_error_forever(int count)
{
    while (1) {
        for (int i = 0; i < count; i++) {
            blink_once(150, 200);
        }
        vTaskDelay(pdMS_TO_TICKS(1500));
    }
}

/* Two slow flashes repeatedly = test successfully completed */

static void blink_complete_forever(void)
{
    while (1) {
        blink_once(400, 300);
        blink_once(400, 2000);
    }
}

/* SD CARD */

static esp_err_t init_sd(sdmmc_card_t **card)
{
    ESP_LOGI(TAG, "Mounting SD card");

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = 10000;

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;

    esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 8,
        .allocation_unit_size = 16 * 1024
    };

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(MOUNT_POINT, &host, &slot_config, &mount_config, card);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "SD mounted successfully");
    return ESP_OK;
}

/* CAMERA */

static esp_err_t init_camera(void)
{
    camera_config_t config = {
        .pin_pwdn = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = 10000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_YUV422,
        .frame_size = FRAMESIZE_QQVGA,
        .jpeg_quality = 12,
        .fb_count = 1,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
        .fb_location = CAMERA_FB_IN_PSRAM
    };

    ESP_LOGI(TAG, "Initializing camera");

    esp_err_t ret = esp_camera_init(&config);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "Camera initialized");
    return ESP_OK;
}

/* RUN DIRECTORY */

static esp_err_t create_run_directory(char *run_directory, size_t size)
{
    char root[64];
    snprintf(root, sizeof(root), "%s/results", MOUNT_POINT);

    ESP_LOGI(TAG, "Checking results directory: %s", root);

    if (mkdir(root, 0775) != 0) {
        if (errno != EEXIST) {
            ESP_LOGE(TAG, "Failed to create %s: errno=%d (%s)", root, errno, strerror(errno));
            return ESP_FAIL;
        }
        ESP_LOGI(TAG, "Results directory already exists");
    } else {
        ESP_LOGI(TAG, "Created results directory");
    }

    for (int run = 1; run <= 999; run++) {
        snprintf(run_directory, size, "%s/run_%03d", root, run);

        if (mkdir(run_directory, 0775) == 0) {
            ESP_LOGI(TAG, "Created run directory: %s", run_directory);
            return ESP_OK;
        }

        if (errno == EEXIST) {
            continue;
        }

        ESP_LOGE(TAG, "Failed to create %s: errno=%d (%s)", run_directory, errno, strerror(errno));
        return ESP_FAIL;
    }

    ESP_LOGE(TAG, "No free run directory numbers available");
    return ESP_FAIL;
}

static esp_err_t write_status(const char *run_directory, const char *status)
{
    char filename[128];
    snprintf(filename, sizeof(filename), "%s/status.txt", run_directory);

    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        ESP_LOGE(TAG, "Failed to open status file: errno=%d (%s)", errno, strerror(errno));
        return ESP_FAIL;
    }

    fprintf(file, "%s\n", status);
    fflush(file);
    fclose(file);

    ESP_LOGI(TAG, "Status changed to: %s", status);
    return ESP_OK;
}

static void save_configuration(const char *run_directory)
{
    char filename[128];

    snprintf(
        filename,
        sizeof(filename),
        "%s/config.txt",
        run_directory
    );

    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        ESP_LOGE(TAG, "Failed to write configuration file");
        return;
    }

    fprintf(
        file,

        "ESP32-CAM Chessboard Localization Benchmark\n"
        "============================================\n"
        "\n"

        "HARDWARE\n"
        "--------\n"
        "Board: ESP32-CAM\n"
        "Processor: ESP32 Xtensa LX6\n"
        "CPU frequency: 240 MHz\n"
        "Flash: 4 MB\n"
        "PSRAM: 2 MB\n"
        "Camera: OV2640\n"
        "\n"

        "CAMERA CONFIGURATION\n"
        "--------------------\n"
        "Pixel format: YUV422\n"
        "Input resolution: 160x120 QQVGA\n"
        "XCLK: 10 MHz\n"
        "Camera framebuffer count: 1\n"
        "Camera framebuffer location: PSRAM\n"
        "DSP input: Y/luminance channel\n"
        "\n"

        "FINAL LOCALIZATION PIPELINE\n"
        "---------------------------\n"
        "1. Y-channel extraction\n"
        "2. Gaussian 3x3 filtering\n"
        "3. Harris corner response\n"
        "4. Harris non-maximum suppression\n"
        "5. Grid direction estimation\n"
        "6. Weighted 1-D projection peak detection\n"
        "7. 9x9 chessboard grid fitting\n"
        "8. Ordered board-corner extraction\n"
        "9. Projective perspective correction\n"
        "\n"

        "Normalized board size: %dx%d\n"
        "Normalized chess square size: %dx%d pixels\n"
        "\n"

        "LEGACY / COMPARISON PIPELINE\n"
        "----------------------------\n"
        "Gaussian + Sobel + threshold + ROI + Hough transform\n"
        "Hough processing retained for benchmark comparison.\n"
        "\n"

        "MEMORY\n"
        "------\n"
        "Camera framebuffer: PSRAM\n"
        "Main DSP image buffers: internal RAM\n"
        "Perspective-warp buffer: internal RAM\n"
        "\n"

        "TEST CONFIGURATION\n"
        "------------------\n"
        "Edge threshold: %d\n"
        "Warm-up frames: %d\n"
        "Measured frames: %d\n"
        "Save every N frames: %d\n"
        "\n",

        BOARD_WARP_SIZE,
        BOARD_WARP_SIZE,

        BOARD_WARP_SIZE / 8,
        BOARD_WARP_SIZE / 8,

        EDGE_THRESHOLD,
        WARMUP_FRAMES,
        TEST_FRAMES,
        SAVE_EVERY_N
    );

    fclose(file);
}

/* DSP */

static esp_err_t init_dsp_buffers(void)
{
    size_t internal_before =heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    ESP_LOGI(TAG, "Free internal RAM before DSP allocation: %u bytes",(unsigned)internal_before);

    /*
     * Main 160x120 DSP buffers.
     */
    gray_buffer = heap_caps_malloc(IMG_PIXELS, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    blur_buffer = heap_caps_malloc(IMG_PIXELS, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    gradient_buffer = heap_caps_malloc(IMG_PIXELS, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    edge_buffer = heap_caps_malloc(IMG_PIXELS, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    if (gray_buffer == NULL ||
        blur_buffer == NULL ||
        gradient_buffer == NULL ||
        edge_buffer == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to allocate main DSP buffers"
        );

        return ESP_ERR_NO_MEM;
    }

    /*
     * 96x96 perspective-corrected
     * chessboard buffer.
     */
    board_warp_buffer =
        heap_caps_malloc(
            BOARD_WARP_PIXELS,
            MALLOC_CAP_INTERNAL |
            MALLOC_CAP_8BIT
        );

    if (board_warp_buffer == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to allocate board warp buffer"
        );

        return ESP_ERR_NO_MEM;
    }

    size_t internal_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

        ESP_LOGI(TAG, "DSP buffers allocated in INTERNAL RAM");

        ESP_LOGI(TAG, "Four 160x120 DSP buffers: %u bytes", (unsigned)(IMG_PIXELS * 4));

        ESP_LOGI(TAG, "Board warp buffer: %u bytes", (unsigned)BOARD_WARP_PIXELS);

        ESP_LOGI(TAG, "Total primary DSP buffer memory: %u bytes", (unsigned)(IMG_PIXELS * 4 + BOARD_WARP_PIXELS));

        ESP_LOGI(TAG, "Free internal RAM after DSP allocation: %u bytes", (unsigned)internal_after);

    return ESP_OK;  
}

static esp_err_t init_hough(void)
{
    hough_accumulator = heap_caps_calloc(HOUGH_THETA_BINS * HOUGH_RHO_BINS, sizeof(uint8_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    line_buffer = heap_caps_malloc(IMG_PIXELS, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    if (hough_accumulator == NULL || line_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate Hough buffers");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "Hough accumulator: %u bytes", (unsigned)(HOUGH_THETA_BINS * HOUGH_RHO_BINS));
    ESP_LOGI(TAG, "Harris/line working buffer: %u bytes INTERNAL RAM", (unsigned)IMG_PIXELS);
    ESP_LOGI(TAG, "Free internal RAM after Hough/work allocation: %u bytes", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    for (int t = 0; t < HOUGH_THETA_BINS; t++) {

        float degrees = (float)(t * HOUGH_THETA_STEP_DEG);
        float radians = degrees * PI_F / 180.0f;

        hough_cos_q15[t] = (int16_t)(cosf(radians) * 32767.0f);
        hough_sin_q15[t] = (int16_t)(sinf(radians) * 32767.0f);
    }

    ESP_LOGI(TAG, "HOUGH initialized");
    ESP_LOGI(TAG, "Hough accumulator size: %u bytes", (unsigned)(HOUGH_THETA_BINS * HOUGH_RHO_BINS));

    return ESP_OK;
}

static void extract_y_channel(const camera_fb_t *fb, uint8_t *output)
{
    const uint8_t *src = fb->buf;

    for (int i = 0; i < IMG_PIXELS; i++) {
        output[i] = src[i * 2];
    }
}

static void gaussian_3x3(const uint8_t *input, uint8_t *output)
{
    for (int x = 0; x < IMG_WIDTH; x++) {
        output[x] = input[x];
        output[(IMG_HEIGHT - 1) * IMG_WIDTH + x] = input[(IMG_HEIGHT - 1) * IMG_WIDTH + x];
    }

    for (int y = 0; y < IMG_HEIGHT; y++) {
        output[y * IMG_WIDTH] = input[y * IMG_WIDTH];
        output[y * IMG_WIDTH + IMG_WIDTH - 1] = input[y * IMG_WIDTH + IMG_WIDTH - 1];
    }

    for (int y = 1; y < IMG_HEIGHT - 1; y++) {
        for (int x = 1; x < IMG_WIDTH - 1; x++) {
            int idx = y * IMG_WIDTH + x;

            int sum = input[idx - IMG_WIDTH - 1] + (input[idx - IMG_WIDTH] << 1) + input[idx - IMG_WIDTH + 1]
                    + (input[idx - 1] << 1) + (input[idx] << 2) + (input[idx + 1] << 1)
                    + input[idx + IMG_WIDTH - 1] + (input[idx + IMG_WIDTH] << 1) + input[idx + IMG_WIDTH + 1];

            output[idx] = (uint8_t)(sum >> 4);
        }
    }
}

static void sobel_filter(const uint8_t *input, uint8_t *output)
{
    memset(output, 0, IMG_PIXELS);

    for (int y = 1; y < IMG_HEIGHT - 1; y++) {
        for (int x = 1; x < IMG_WIDTH - 1; x++) {
            int idx = y * IMG_WIDTH + x;

            int gx = -input[idx - IMG_WIDTH - 1] + input[idx - IMG_WIDTH + 1]
                    - (input[idx - 1] << 1) + (input[idx + 1] << 1)
                    - input[idx + IMG_WIDTH - 1] + input[idx + IMG_WIDTH + 1];

            int gy = -input[idx - IMG_WIDTH - 1] - (input[idx - IMG_WIDTH] << 1) - input[idx - IMG_WIDTH + 1]
                    + input[idx + IMG_WIDTH - 1] + (input[idx + IMG_WIDTH] << 1) + input[idx + IMG_WIDTH + 1];

            int magnitude = abs(gx) + abs(gy);

            if (magnitude > 255) {
                magnitude = 255;
            }

            output[idx] = (uint8_t)magnitude;
        }
    }
}

static void threshold_edges(const uint8_t *gradient, uint8_t *edges)
{
    for (int i = 0; i < IMG_PIXELS; i++) {
        edges[i] = gradient[i] >= EDGE_THRESHOLD ? 255 : 0;
    }
}

static void detect_edge_roi(const uint8_t *edges, edge_roi_t *roi)
{
    uint8_t dense[ROI_TOTAL_BLOCKS];
    uint8_t visited[ROI_TOTAL_BLOCKS];
    uint8_t best_component[ROI_TOTAL_BLOCKS];
    uint8_t expanded_mask[ROI_TOTAL_BLOCKS];

    int queue[ROI_TOTAL_BLOCKS];

    memset(dense, 0, sizeof(dense));
    memset(visited, 0, sizeof(visited));
    memset(best_component, 0, sizeof(best_component));
    memset(expanded_mask, 0, sizeof(expanded_mask));
    memset(roi_block_mask, 0, sizeof(roi_block_mask));

    for (int by = 0; by < ROI_BLOCK_ROWS; by++) {

        for (int bx = 0; bx < ROI_BLOCK_COLS; bx++) {

            int edge_count = 0;

            int x_start = bx * ROI_BLOCK_SIZE;
            int y_start = by * ROI_BLOCK_SIZE;

            for (int y = y_start; y < y_start + ROI_BLOCK_SIZE; y++) {

                for (int x = x_start; x < x_start + ROI_BLOCK_SIZE; x++) {

                    if (edges[y * IMG_WIDTH + x] != 0) {
                        edge_count++;
                    }
                }
            }

            if (edge_count >= ROI_DENSE_EDGE_COUNT) {
                dense[by * ROI_BLOCK_COLS + bx] = 1;
            }
        }
    }

    int best_count = 0;

    for (int start = 0; start < ROI_TOTAL_BLOCKS; start++) {

        if (dense[start] == 0 || visited[start] != 0) {
            continue;
        }

        int head = 0;
        int tail = 0;

        queue[tail++] = start;
        visited[start] = 1;

        while (head < tail) {

            int index = queue[head++];

            int bx = index % ROI_BLOCK_COLS;
            int by = index / ROI_BLOCK_COLS;

            const int dx[4] = {1, -1, 0, 0};
            const int dy[4] = {0, 0, 1, -1};

            for (int n = 0; n < 4; n++) {

                int nx = bx + dx[n];
                int ny = by + dy[n];

                if (nx < 0 || nx >= ROI_BLOCK_COLS ||
                    ny < 0 || ny >= ROI_BLOCK_ROWS) {
                    continue;
                }

                int neighbour =
                    ny * ROI_BLOCK_COLS + nx;

                if (dense[neighbour] == 0 ||
                    visited[neighbour] != 0) {
                    continue;
                }

                visited[neighbour] = 1;
                queue[tail++] = neighbour;
            }
        }

        if (tail > best_count) {

            best_count = tail;

            memset(
                best_component,
                0,
                sizeof(best_component)
            );

            for (int i = 0; i < tail; i++) {
                best_component[queue[i]] = 1;
            }
        }
    }

    if (best_count < ROI_MIN_BLOCKS) {

        for (int i = 0; i < ROI_TOTAL_BLOCKS; i++) {
            roi_block_mask[i] = 1;
        }

        roi->x0 = 0;
        roi->y0 = 0;
        roi->x1 = IMG_WIDTH;
        roi->y1 = IMG_HEIGHT;

        roi->dense_blocks = best_count;
        roi->valid = 0;

        return;
    }

    for (int by = 0; by < ROI_BLOCK_ROWS; by++) {

        for (int bx = 0; bx < ROI_BLOCK_COLS; bx++) {

            int index =
                by * ROI_BLOCK_COLS + bx;

            if (best_component[index] == 0) {
                continue;
            }

            for (int dy = -ROI_EXPAND_BLOCKS; dy <= ROI_EXPAND_BLOCKS; dy++) {

                for (int dx = -ROI_EXPAND_BLOCKS; dx <= ROI_EXPAND_BLOCKS; dx++) {

                    int nx = bx + dx;
                    int ny = by + dy;

                    if (nx < 0 || nx >= ROI_BLOCK_COLS ||
                        ny < 0 || ny >= ROI_BLOCK_ROWS) {
                        continue;
                    }

                    expanded_mask[
                        ny * ROI_BLOCK_COLS + nx
                    ] = 1;
                }
            }
        }
    }

    memcpy(roi_block_mask, expanded_mask, sizeof(roi_block_mask));

    int min_bx = ROI_BLOCK_COLS;
    int min_by = ROI_BLOCK_ROWS;

    int max_bx = -1;
    int max_by = -1;

    for (int by = 0; by < ROI_BLOCK_ROWS; by++) {

        for (int bx = 0; bx < ROI_BLOCK_COLS; bx++) {

            if (roi_block_mask[by * ROI_BLOCK_COLS + bx] == 0) {
                continue;
            }

            if (bx < min_bx) {
                min_bx = bx;
            }

            if (bx > max_bx) {
                max_bx = bx;
            }

            if (by < min_by) {
                min_by = by;
            }

            if (by > max_by) {
                max_by = by;
            }
        }
    }

    if (max_bx < 0 || max_by < 0) {

        for (int i = 0; i < ROI_TOTAL_BLOCKS; i++) {
            roi_block_mask[i] = 1;
        }

        roi->x0 = 0;
        roi->y0 = 0;
        roi->x1 = IMG_WIDTH;
        roi->y1 = IMG_HEIGHT;

        roi->dense_blocks = best_count;
        roi->valid = 0;

        return;
    }

    roi->x0 = min_bx * ROI_BLOCK_SIZE;
    roi->y0 = min_by * ROI_BLOCK_SIZE;

    roi->x1 = (max_bx + 1) * ROI_BLOCK_SIZE;
    roi->y1 = (max_by + 1) * ROI_BLOCK_SIZE;

    roi->dense_blocks = best_count;
    roi->valid = 1;
}

static void hough_vote_masked(const uint8_t *edges)
{
    memset(
        hough_accumulator,
        0,
        HOUGH_THETA_BINS * HOUGH_RHO_BINS
    );

    for (int by = 0; by < ROI_BLOCK_ROWS; by++) {
        for (int bx = 0; bx < ROI_BLOCK_COLS; bx++) {

            int block_index =
                by * ROI_BLOCK_COLS + bx;

            if (roi_block_mask[block_index] == 0) {
                continue;
            }

            int x_start = bx * ROI_BLOCK_SIZE;
            int y_start = by * ROI_BLOCK_SIZE;

            int x_end = x_start + ROI_BLOCK_SIZE;
            int y_end = y_start + ROI_BLOCK_SIZE;

            for (int y = y_start; y < y_end; y++) {
                for (int x = x_start; x < x_end; x++) {

                    int index =
                        y * IMG_WIDTH + x;

                    if (edges[index] == 0) {
                        continue;
                    }

                    for (int t = 0; t < HOUGH_THETA_BINS; t++) {

                        int32_t rho_q15 =
                            ((int32_t)x * hough_cos_q15[t]) +
                            ((int32_t)y * hough_sin_q15[t]);

                        int rho =
                            rho_q15 >> 15;

                        int rho_index =
                            rho + HOUGH_RHO_MAX;

                        if (rho_index < 0 ||
                            rho_index >= HOUGH_RHO_BINS) {
                            continue;
                        }

                        int accumulator_index =
                            t * HOUGH_RHO_BINS +
                            rho_index;

                        if (hough_accumulator[accumulator_index] < 255) {
                            hough_accumulator[accumulator_index]++;
                        }
                    }
                }
            }
        }
    }
}

static void suppress_hough_peak(int theta_index, int rho_index)
{
    for (int dt = -HOUGH_SUPPRESS_THETA; dt <= HOUGH_SUPPRESS_THETA; dt++) {
        
        int t = theta_index + dt;

        while (t < 0) {
            t += HOUGH_THETA_BINS;
        }

        while (t >= HOUGH_THETA_BINS) {
            t -= HOUGH_THETA_BINS;
        }

        for (int dr = -HOUGH_SUPPRESS_RHO; dr <= HOUGH_SUPPRESS_RHO; dr++) {
            
            int r = rho_index + dr;

            if (r < 0 || r >= HOUGH_RHO_BINS) {
                continue;
            }

            hough_accumulator[t * HOUGH_RHO_BINS + r] = 0;
        }
    }
}

static int find_hough_lines(void)
{
    int line_count = 0;

    for (int line = 0; line < HOUGH_MAX_LINES; line++) {

        uint8_t best_votes = 0;
        int best_theta = -1;
        int best_rho_index = -1;

        for (int t = 0; t < HOUGH_THETA_BINS; t++) {

            for (int r = 0; r < HOUGH_RHO_BINS; r++) {

                uint8_t votes =
                    hough_accumulator[t * HOUGH_RHO_BINS + r];

                if (votes > best_votes) {

                    best_votes = votes;
                    best_theta = t;
                    best_rho_index = r;
                }
            }
        }

        if (best_theta < 0 || best_votes < HOUGH_MIN_VOTES) {
            break;
        }

        hough_lines[line_count].theta_index = best_theta;

        hough_lines[line_count].rho =
            best_rho_index - HOUGH_RHO_MAX;

        hough_lines[line_count].votes =
            best_votes;

        line_count++;

        suppress_hough_peak(best_theta, best_rho_index);
    }

    return line_count;
}

static int theta_distance(int a, int b)
{
    int d = abs(a - b);

    if (d > HOUGH_THETA_BINS / 2) {
        d = HOUGH_THETA_BINS - d;
    }

    return d;
}

static int canonical_line_position(const hough_line_t *line, int family_theta)
{
    int64_t dot =
        ((int64_t)hough_cos_q15[line->theta_index] * hough_cos_q15[family_theta]) +
        ((int64_t)hough_sin_q15[line->theta_index] * hough_sin_q15[family_theta]);

    if (dot < 0) {
        return -line->rho;
    }

    return line->rho;
}

static int select_outer_family_lines(hough_line_t *lines, int count, int family_theta, hough_line_t *min_line, hough_line_t *max_line)
{
    if (count < 2) {
        return 0;
    }

    int min_position = INT_MAX;
    int max_position = INT_MIN;

    int min_index = -1;
    int max_index = -1;

    for (int i = 0; i < count; i++) {
        int position = canonical_line_position(&lines[i], family_theta);

        if (position < min_position) {
            min_position = position;
            min_index = i;
        }

        if (position > max_position) {
            max_position = position;
            max_index = i;
        }
    }

    if (min_index < 0 || max_index < 0) {
        return 0;
    }

    if ((max_position - min_position) < BOARD_MIN_FAMILY_SPAN) {
        return 0;
    }

    *min_line = lines[min_index];
    *max_line = lines[max_index];

    return 1;
}

static int intersect_hough_lines(const hough_line_t *a, const hough_line_t *b, point_f_t *point)
{
    float c1 = (float)hough_cos_q15[a->theta_index] / 32767.0f;
    float s1 = (float)hough_sin_q15[a->theta_index] / 32767.0f;

    float c2 = (float)hough_cos_q15[b->theta_index] / 32767.0f;
    float s2 = (float)hough_sin_q15[b->theta_index] / 32767.0f;

    float determinant = c1 * s2 - s1 * c2;

    if (fabsf(determinant) < 0.05f) {
        return 0;
    }

    point->x = ((float)a->rho * s2 - (float)b->rho * s1) / determinant;
    point->y = (c1 * (float)b->rho - c2 * (float)a->rho) / determinant;

    return 1;
}

static float board_quad_area(const board_detection_t *board)
{
    float area = 0.0f;

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        area += board->corner[i].x * board->corner[j].y;
        area -= board->corner[j].x * board->corner[i].y;
    }

    if (area < 0.0f) {
        area = -area;
    }

    return area * 0.5f;
}

static int wrap_theta_index(int theta)
{
    while (theta < 0) {
        theta += HOUGH_THETA_BINS;
    }

    while (theta >= HOUGH_THETA_BINS) {
        theta -= HOUGH_THETA_BINS;
    }

    return theta;
}

static int find_family_rho_peaks(int family_theta, hough_line_t *peaks)
{
    static uint8_t rho_profile[HOUGH_RHO_BINS];
    static uint8_t rho_theta[HOUGH_RHO_BINS];
    static uint8_t work[HOUGH_RHO_BINS];

    memset(rho_profile, 0, sizeof(rho_profile));
    memset(rho_theta, 0, sizeof(rho_theta));

    for (int r = 0; r < HOUGH_RHO_BINS; r++) {
        uint8_t best_vote = 0;
        int best_theta = family_theta;

        for (int dt = -FAMILY_ANGLE_WINDOW; dt <= FAMILY_ANGLE_WINDOW; dt++) {

            int t = wrap_theta_index(family_theta + dt);

            uint8_t votes =
                hough_accumulator[t * HOUGH_RHO_BINS + r];

            if (votes > best_vote) {
                best_vote = votes;
                best_theta = t;
            }
        }

        rho_profile[r] = best_vote;
        rho_theta[r] = (uint8_t)best_theta;
    }

    memcpy(work, rho_profile, sizeof(work));

    int count = 0;

    while (count < OUTER_LINE_MAX) {
        int best_r = -1;
        uint8_t best_vote = 0;

        for (int r = 0; r < HOUGH_RHO_BINS; r++) {
            if (work[r] > best_vote) {
                best_vote = work[r];
                best_r = r;
            }
        }

        if (best_r < 0 ||
            best_vote < OUTER_MIN_PEAK_VOTES) {
            break;
        }

        peaks[count].theta_index = rho_theta[best_r];
        peaks[count].rho = best_r - HOUGH_RHO_MAX;
        peaks[count].votes = best_vote;

        count++;

        int r0 = best_r - OUTER_RHO_NMS;
        int r1 = best_r + OUTER_RHO_NMS;

        if (r0 < 0) {
            r0 = 0;
        }

        if (r1 >= HOUGH_RHO_BINS) {
            r1 = HOUGH_RHO_BINS - 1;
        }

        for (int r = r0; r <= r1; r++) {
            work[r] = 0;
        }
    }

    return count;
}

static int build_outer_line_pairs(int family_theta, line_pair_candidate_t *pairs)
{
    static hough_line_t peaks[OUTER_LINE_MAX];

    int peak_count =
        find_family_rho_peaks(
            family_theta,
            peaks);

    if (peak_count < 2) {
        return 0;
    }

    int pair_count = 0;

    for (int i = 0; i < peak_count; i++) {
        for (int j = i + 1; j < peak_count; j++) {

            int p1 =
                canonical_line_position(
                    &peaks[i],
                    family_theta);

            int p2 =
                canonical_line_position(
                    &peaks[j],
                    family_theta);

            int span = abs(p2 - p1);

            if (span < BOARD_MIN_FAMILY_SPAN) {
                continue;
            }

            line_pair_candidate_t candidate;

            if (p1 < p2) {
                candidate.first = peaks[i];
                candidate.last = peaks[j];
            } else {
                candidate.first = peaks[j];
                candidate.last = peaks[i];
            }

            candidate.span = span;

            candidate.score =
                (uint32_t)span * 1000U +
                (uint32_t)(peaks[i].votes +
                           peaks[j].votes) * 20U;

            if (pair_count < OUTER_PAIR_MAX) {
                pairs[pair_count++] = candidate;
            } else {
                int weakest = 0;

                for (int k = 1; k < OUTER_PAIR_MAX; k++) {
                    if (pairs[k].score <
                        pairs[weakest].score) {
                        weakest = k;
                    }
                }

                if (candidate.score >
                    pairs[weakest].score) {
                    pairs[weakest] = candidate;
                }
            }
        }
    }

    return pair_count;
}

static int point_inside_board_quad(const board_detection_t *board, float x, float y)
{
    int positive = 0;
    int negative = 0;

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        float ax = board->corner[i].x;
        float ay = board->corner[i].y;

        float bx = board->corner[j].x;
        float by = board->corner[j].y;

        float cross =
            (bx - ax) * (y - ay) -
            (by - ay) * (x - ax);

        if (cross > 0.0f) {
            positive = 1;
        }

        if (cross < 0.0f) {
            negative = 1;
        }

        if (positive && negative) {
            return 0;
        }
    }

    return 1;
}

static void board_roi_metrics(const board_detection_t *board, int *coverage, int *precision)
{
    int roi_total = 0;
    int roi_inside = 0;
    int quad_total = 0;

    for (int by = 0; by < ROI_BLOCK_ROWS; by++) {
        for (int bx = 0; bx < ROI_BLOCK_COLS; bx++) {
            int block_index = by * ROI_BLOCK_COLS + bx;

            float x =
                bx * ROI_BLOCK_SIZE +
                ROI_BLOCK_SIZE * 0.5f;

            float y =
                by * ROI_BLOCK_SIZE +
                ROI_BLOCK_SIZE * 0.5f;

            int in_roi =
                roi_block_mask[block_index] != 0;

            int in_quad =
                point_inside_board_quad(
                    board,
                    x,
                    y);

            if (in_roi) {
                roi_total++;
            }

            if (in_quad) {
                quad_total++;
            }

            if (in_roi && in_quad) {
                roi_inside++;
            }
        }
    }

    if (roi_total > 0) {
        *coverage =
            (roi_inside * 100) / roi_total;
    } else {
        *coverage = 0;
    }

    if (quad_total > 0) {
        *precision =
            (roi_inside * 100) / quad_total;
    } else {
        *precision = 0;
    }
}


static int find_chess_grid_candidates(int family_theta, grid_candidate_t *candidates)
{
    uint8_t rho_profile[HOUGH_RHO_BINS];
    uint8_t rho_theta[HOUGH_RHO_BINS];

    memset(rho_profile, 0, sizeof(rho_profile));
    memset(rho_theta, 0, sizeof(rho_theta));

    for (int r = 0; r < HOUGH_RHO_BINS; r++) {
        uint8_t best_vote = 0;
        int best_theta = family_theta;

        for (int dt = -FAMILY_ANGLE_WINDOW; dt <= FAMILY_ANGLE_WINDOW; dt++) {

            int t = wrap_theta_index(family_theta + dt);

            uint8_t votes =
                hough_accumulator[t * HOUGH_RHO_BINS + r];

            if (votes > best_vote) {
                best_vote = votes;
                best_theta = t;
            }
        }

        rho_profile[r] = best_vote;
        rho_theta[r] = (uint8_t)best_theta;
    }

    int candidate_count = 0;

    for (int spacing = GRID_MIN_SPACING; spacing <= GRID_MAX_SPACING; spacing++) {

        int span = spacing * 8;

        for (int start = 0; start + span < HOUGH_RHO_BINS; start++) {

            uint32_t score = 0;
            int matched = 0;

            for (int grid = 0; grid < 9; grid++) {
                int expected = start + grid * spacing;
                uint8_t peak = 0;

                for (int dr = -GRID_RHO_TOLERANCE; dr <= GRID_RHO_TOLERANCE; dr++) {

                    int r = expected + dr;

                    if (r < 0 || r >= HOUGH_RHO_BINS) {
                        continue;
                    }

                    if (rho_profile[r] > peak) {
                        peak = rho_profile[r];
                    }
                }

                if (peak >= GRID_MIN_PEAK_VOTES) {
                    matched++;
                    score += (uint32_t)peak * (uint32_t)peak;
                }
            }

            if (matched < GRID_MIN_MATCHED_LINES) {
                continue;
            }

            score += (uint32_t)matched * 25000U;

            int expected_first = start;
            int expected_last = start + 8 * spacing;

            int first_index = expected_first;
            int last_index = expected_last;

            uint8_t first_votes = 0;
            uint8_t last_votes = 0;

            for (int dr = -GRID_RHO_TOLERANCE; dr <= GRID_RHO_TOLERANCE; dr++) {

                int r = expected_first + dr;

                if (r >= 0 &&
                    r < HOUGH_RHO_BINS &&
                    rho_profile[r] > first_votes) {

                    first_votes = rho_profile[r];
                    first_index = r;
                }
            }

            for (int dr = -GRID_RHO_TOLERANCE; dr <= GRID_RHO_TOLERANCE; dr++) {

                int r = expected_last + dr;

                if (r >= 0 &&
                    r < HOUGH_RHO_BINS &&
                    rho_profile[r] > last_votes) {

                    last_votes = rho_profile[r];
                    last_index = r;
                }
            }

            grid_candidate_t candidate;

            candidate.first.theta_index =
                first_votes > 0
                ? rho_theta[first_index]
                : family_theta;

            candidate.first.rho =
                first_index - HOUGH_RHO_MAX;

            candidate.first.votes = first_votes;

            candidate.last.theta_index =
                last_votes > 0
                ? rho_theta[last_index]
                : family_theta;

            candidate.last.rho =
                last_index - HOUGH_RHO_MAX;

            candidate.last.votes = last_votes;

            candidate.spacing = spacing;
            candidate.matched = matched;
            candidate.score = score;

            int duplicate = -1;

            for (int i = 0; i < candidate_count; i++) {
                int first_diff =
                    abs(candidate.first.rho -
                        candidates[i].first.rho);

                int last_diff =
                    abs(candidate.last.rho -
                        candidates[i].last.rho);

                if (first_diff <= GRID_DUPLICATE_RHO &&
                    last_diff <= GRID_DUPLICATE_RHO) {
                    duplicate = i;
                    break;
                }
            }

            if (duplicate >= 0) {
                if (candidate.score >
                    candidates[duplicate].score) {
                    candidates[duplicate] = candidate;
                }

                continue;
            }

            if (candidate_count < GRID_MAX_CANDIDATES) {
                candidates[candidate_count++] = candidate;
            } else {
                int weakest = 0;

                for (int i = 1; i < GRID_MAX_CANDIDATES; i++) {

                    if (candidates[i].score <
                        candidates[weakest].score) {
                        weakest = i;
                    }
                }

                if (candidate.score >
                    candidates[weakest].score) {
                    candidates[weakest] = candidate;
                }
            }
        }
    }

    return candidate_count;
}

static int detect_board_corners(int family_a_theta, int family_b_theta, int family_a_count, int family_b_count, board_detection_t *board)
{
    memset(board, 0, sizeof(board_detection_t));

    if (family_a_count < 2 ||
        family_b_count < 2) {
        return 0;
    }

    static line_pair_candidate_t a_pairs[OUTER_PAIR_MAX];
    static line_pair_candidate_t b_pairs[OUTER_PAIR_MAX];

    int a_count =
        build_outer_line_pairs(
            family_a_theta,
            a_pairs);

    int b_count =
        build_outer_line_pairs(
            family_b_theta,
            b_pairs);

    if (a_count == 0 || b_count == 0) {
        return 0;
    }

    float best_score = -1.0f;
    board_detection_t best_board;

    memset(&best_board, 0, sizeof(best_board));

    for (int a = 0; a < a_count; a++) {
        for (int b = 0; b < b_count; b++) {

            board_detection_t test;

            memset(&test, 0, sizeof(test));

            test.a_min = a_pairs[a].first;
            test.a_max = a_pairs[a].last;

            test.b_min = b_pairs[b].first;
            test.b_max = b_pairs[b].last;

            if (!intersect_hough_lines(
                    &test.a_min,
                    &test.b_min,
                    &test.corner[0])) {
                continue;
            }

            if (!intersect_hough_lines(
                    &test.a_min,
                    &test.b_max,
                    &test.corner[1])) {
                continue;
            }

            if (!intersect_hough_lines(
                    &test.a_max,
                    &test.b_max,
                    &test.corner[2])) {
                continue;
            }

            if (!intersect_hough_lines(
                    &test.a_max,
                    &test.b_min,
                    &test.corner[3])) {
                continue;
            }

            int valid = 1;

            for (int i = 0; i < 4; i++) {
                if (test.corner[i].x <
                        -BOARD_CORNER_MARGIN ||
                    test.corner[i].x >
                        IMG_WIDTH + BOARD_CORNER_MARGIN ||
                    test.corner[i].y <
                        -BOARD_CORNER_MARGIN ||
                    test.corner[i].y >
                        IMG_HEIGHT + BOARD_CORNER_MARGIN) {

                    valid = 0;
                    break;
                }
            }

            if (!valid) {
                continue;
            }

            float area =
                board_quad_area(&test);

            if (area < BOARD_MIN_AREA) {
                continue;
            }
            
            int coverage = 0;
            int precision = 0;

            board_roi_metrics(
                &test,
                &coverage,
                &precision
            );

            if (coverage < BOARD_MIN_ROI_COVERAGE) {
                continue;
            }

            if (precision < BOARD_MIN_ROI_PRECISION) {
                continue;
            }

            float support = (float)a_pairs[a].score + (float)b_pairs[b].score;

            float roi_fit =
                (float)coverage *
                (float)precision;

            float score =
                roi_fit * 1000.0f +
                support * 0.02f +
                area * 0.1f;

            if (score > best_score) {
                best_score = score;
                best_board = test;
                best_board.valid = 1;
            }
        }
    }

    if (best_score < 0.0f) {
        return 0;
    }

    *board = best_board;

    return 1;
}


static void draw_segment(uint8_t *buffer, int x0, int y0, int x1, int y1, uint8_t value)
{
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;

    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;

    int error = dx + dy;

    while (1) {
        if (x0 >= 0 && x0 < IMG_WIDTH && y0 >= 0 && y0 < IMG_HEIGHT) {
            buffer[y0 * IMG_WIDTH + x0] = value;
        }

        if (x0 == x1 && y0 == y1) {
            break;
        }

        int e2 = 2 * error;

        if (e2 >= dy) {
            error += dy;
            x0 += sx;
        }

        if (e2 <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

static void render_board_outline(const board_detection_t *board)
{
    memcpy(line_buffer, gray_buffer, IMG_PIXELS);

    if (!board->valid) {
        return;
    }

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        int x0 = (int)(board->corner[i].x + 0.5f);
        int y0 = (int)(board->corner[i].y + 0.5f);

        int x1 = (int)(board->corner[j].x + 0.5f);
        int y1 = (int)(board->corner[j].y + 0.5f);

        draw_segment(
            line_buffer,
            x0,
            y0,
            x1,
            y1,
            255
        );
    }

    for (int i = 0; i < 4; i++) {
        int cx = (int)(board->corner[i].x + 0.5f);
        int cy = (int)(board->corner[i].y + 0.5f);

        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {

                int x = cx + dx;
                int y = cy + dy;

                if (x >= 0 && x < IMG_WIDTH &&
                    y >= 0 && y < IMG_HEIGHT) {
                    line_buffer[y * IMG_WIDTH + x] = 255;
                }
            }
        }
    }
}

static void calculate_theta_scores(void)
{
    memset(theta_scores, 0, sizeof(theta_scores));

    for (int t = 0; t < HOUGH_THETA_BINS; t++) {

        uint32_t score = 0;

        for (int r = 0; r < HOUGH_RHO_BINS; r++) {

            uint8_t votes = hough_accumulator[t * HOUGH_RHO_BINS + r];

            if (votes >= HOUGH_MIN_VOTES) {
                score += (uint32_t)votes * (uint32_t)votes;
            }
        }

        theta_scores[t] = score;
    }
}

static uint64_t get_theta_band_score(int center)
{
    uint64_t score = 0;

    for (int dt = -FAMILY_SCORE_RADIUS; dt <= FAMILY_SCORE_RADIUS; dt++) {

        int t = center + dt;

        while (t < 0) {
            t += HOUGH_THETA_BINS;
        }

        while (t >= HOUGH_THETA_BINS) {
            t -= HOUGH_THETA_BINS;
        }

        score += theta_scores[t];
    }

    return score;
}

static int find_best_theta_near(int center_theta, int radius)
{
    int best_theta = -1;
    uint32_t best_score = 0;

    for (int d = -radius; d <= radius; d++) {
        int t = center_theta + d;

        while (t < 0) {
            t += HOUGH_THETA_BINS;
        }

        while (t >= HOUGH_THETA_BINS) {
            t -= HOUGH_THETA_BINS;
        }

        uint32_t score = get_theta_band_score(t);

        if (score > best_score) {
            best_score = score;
            best_theta = t;
        }
    }

    return best_theta;
}

static int find_best_theta_global(int exclude_theta)
{
    int best_theta = -1;
    uint32_t best_score = 0;

    for (int t = 0; t < HOUGH_THETA_BINS; t++) {
        if (exclude_theta >= 0 &&
            theta_distance(t, exclude_theta) <
            FAMILY_MIN_SEPARATION_BINS) {
            continue;
        }

        uint32_t score = get_theta_band_score(t);

        if (score > best_score) {
            best_score = score;
            best_theta = t;
        }
    }

    return best_theta;
}

static int family_line_is_duplicate(hough_line_t *lines, int count, int theta, int rho)
{
    for (int i = 0; i < count; i++) {

        int theta_diff = theta_distance(theta, lines[i].theta_index);
        int rho_diff = abs(rho - lines[i].rho);

        if (theta_diff <= HOUGH_SUPPRESS_THETA && rho_diff <= HOUGH_SUPPRESS_RHO) {
            return 1;
        }
    }

    return 0;
}

static int extract_family_lines(int center_theta, hough_line_t *output)
{
    int count = 0;

    while (count < FAMILY_MAX_LINES) {

        uint8_t best_votes = 0;
        int best_theta = -1;
        int best_rho = 0;

        for (int t = 0; t < HOUGH_THETA_BINS; t++) {

            if (theta_distance(t, center_theta) > FAMILY_ANGLE_WINDOW) {
                continue;
            }

            for (int r = 0; r < HOUGH_RHO_BINS; r++) {

                uint8_t votes = hough_accumulator[t * HOUGH_RHO_BINS + r];

                if (votes < HOUGH_MIN_VOTES || votes <= best_votes) {
                    continue;
                }

                int rho = r - HOUGH_RHO_MAX;

                if (family_line_is_duplicate(output, count, t, rho)) {
                    continue;
                }

                best_votes = votes;
                best_theta = t;
                best_rho = rho;
            }
        }

        if (best_theta < 0) {
            break;
        }

        output[count].theta_index = best_theta;
        output[count].rho = best_rho;
        output[count].votes = best_votes;

        count++;
    }

    return count;
}

static void detect_hough_families(int *family_a_theta, int *family_b_theta, int *family_a_count, int *family_b_count) 
{
    *family_a_theta = -1;
    *family_b_theta = -1;
    *family_a_count = 0;
    *family_b_count = 0;

    calculate_theta_scores();

    /*
     * If tracking is already established, first look only
     * close to the previous two family orientations.
     */
    if (tracked_family_a_theta >= 0 &&
        tracked_family_b_theta >= 0) {

        int candidate_a = find_best_theta_near(
            tracked_family_a_theta,
            FAMILY_TRACK_MAX_JUMP_BINS
        );

        int candidate_b = find_best_theta_near(
            tracked_family_b_theta,
            FAMILY_TRACK_MAX_JUMP_BINS
        );

        if (candidate_a >= 0 &&
            candidate_b >= 0 &&
            theta_distance(candidate_a, candidate_b) >=
                FAMILY_MIN_SEPARATION_BINS) {

            int a_count = extract_family_lines(
                candidate_a,
                family_a_lines
            );

            int b_count = extract_family_lines(
                candidate_b,
                family_b_lines
            );

            if (a_count >= FAMILY_TRACK_MIN_LINES &&
                b_count >= FAMILY_TRACK_MIN_LINES) {

                tracked_family_a_theta = candidate_a;
                tracked_family_b_theta = candidate_b;
                family_track_misses = 0;

                *family_a_theta = candidate_a;
                *family_b_theta = candidate_b;
                *family_a_count = a_count;
                *family_b_count = b_count;

                return;
            }
        }

        /*
         * A frame did not contain enough evidence close to
         * the tracked orientations.
         *
         * Do NOT immediately jump to some unrelated strong
         * diagonal. Count it as a miss instead.
         */
        family_track_misses++;

        if (family_track_misses < FAMILY_TRACK_RESET_MISSES) {
            int a_count = extract_family_lines(
                tracked_family_a_theta,
                family_a_lines
            );

            int b_count = extract_family_lines(
                tracked_family_b_theta,
                family_b_lines
            );

            if (a_count >= 2 && b_count >= 2) {
                *family_a_theta = tracked_family_a_theta;
                *family_b_theta = tracked_family_b_theta;
                *family_a_count = a_count;
                *family_b_count = b_count;
            }

            return;
        }

        /*
         * Too many consecutive misses.
         * Drop the tracker and reacquire globally.
         */
        tracked_family_a_theta = -1;
        tracked_family_b_theta = -1;
        family_track_misses = 0;
    }

    /*
     * Global acquisition.
     */
    int candidate_a =
        find_best_theta_global(-1);

    if (candidate_a < 0) {
        return;
    }

    int candidate_b =
        find_best_theta_global(candidate_a);

    if (candidate_b < 0) {
        return;
    }

    int a_count =
        extract_family_lines(
            candidate_a,
            family_a_lines
        );

    int b_count =
        extract_family_lines(
            candidate_b,
            family_b_lines
        );

    *family_a_theta = candidate_a;
    *family_b_theta = candidate_b;
    *family_a_count = a_count;
    *family_b_count = b_count;

    /*
     * Only lock tracking when both families have enough
     * actual Hough lines.
     */
    if (a_count >= FAMILY_TRACK_MIN_LINES &&
        b_count >= FAMILY_TRACK_MIN_LINES) {

        tracked_family_a_theta = candidate_a;
        tracked_family_b_theta = candidate_b;
        family_track_misses = 0;
    }
}

static void draw_hough_line(uint8_t *buffer, int theta_index, int target_rho, uint8_t value)
{
    for (int y = 0; y < IMG_HEIGHT; y++) {

        for (int x = 0; x < IMG_WIDTH; x++) {

            int32_t rho_q15 =
                ((int32_t)x * hough_cos_q15[theta_index]) +
                ((int32_t)y * hough_sin_q15[theta_index]);

            int rho = rho_q15 >> 15;

            if (abs(rho - target_rho) <= 1) {
                buffer[y * IMG_WIDTH + x] = value;
            }
        }
    }
}

static void render_hough_families(int family_a_count, int family_b_count)
{
    for (int i = 0; i < IMG_PIXELS; i++) {
        line_buffer[i] = gray_buffer[i] >> 2;
    }

    for (int i = 0; i < family_a_count; i++) {
        draw_hough_line(line_buffer, family_a_lines[i].theta_index, family_a_lines[i].rho, 160);
    }

    for (int i = 0; i < family_b_count; i++) {
        draw_hough_line(line_buffer, family_b_lines[i].theta_index, family_b_lines[i].rho, 255);
    }
}

static void render_hough_lines(int line_count)
{
    memcpy(line_buffer, edge_buffer, IMG_PIXELS);

    for (int line = 0; line < line_count; line++) {

        int theta_index =
            hough_lines[line].theta_index;

        int target_rho =
            hough_lines[line].rho;

        for (int y = 0; y < IMG_HEIGHT; y++) {

            for (int x = 0; x < IMG_WIDTH; x++) {

                int32_t rho_q15 =
                    ((int32_t)x * hough_cos_q15[theta_index]) +
                    ((int32_t)y * hough_sin_q15[theta_index]);

                int rho =
                    rho_q15 >> 15;

                int difference =
                    rho - target_rho;

                if (difference < 0) {
                    difference = -difference;
                }

                if (difference <= 1) {
                    line_buffer[y * IMG_WIDTH + x] = 128;
                }
            }
        }
    }
}

static void render_roi(const edge_roi_t *roi)
{
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            int bx = x / ROI_BLOCK_SIZE;
            int by = y / ROI_BLOCK_SIZE;
            int block_index = by * ROI_BLOCK_COLS + bx;
            int pixel_index = y * IMG_WIDTH + x;

            if (roi_block_mask[block_index] != 0) {
                line_buffer[pixel_index] = gray_buffer[pixel_index];
            } else {
                line_buffer[pixel_index] = gray_buffer[pixel_index] >> 2;
            }
        }
    }

    int left = roi->x0;
    int right = roi->x1 - 1;
    int top = roi->y0;
    int bottom = roi->y1 - 1;

    for (int x = left; x <= right; x++) {
        line_buffer[top * IMG_WIDTH + x] = 255;
        line_buffer[bottom * IMG_WIDTH + x] = 255;
    }

    for (int y = top; y <= bottom; y++) {
        line_buffer[y * IMG_WIDTH + left] = 255;
        line_buffer[y * IMG_WIDTH + right] = 255;
    }
}

static esp_err_t save_gray_pgm(const char *run_directory, const char *prefix, int frame_number, const uint8_t *buffer, int width, int height)
{
    char filename[160];
    snprintf(filename, sizeof(filename), "%s/%s%03d.pgm", run_directory, prefix, frame_number);

    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        ESP_LOGE(TAG, "Failed to open %s", filename);
        return ESP_FAIL;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);

    size_t expected = width * height;
    size_t written = fwrite(buffer, 1, expected, file);

    fflush(file);
    fclose(file);

    if (written != expected) {
        ESP_LOGE(TAG, "Incomplete PGM write: %s", filename);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Saved %s", filename);
    return ESP_OK;
}

static void harris_sobel_at(const uint8_t *image, int x, int y, int *gx, int *gy)
{
    int tl = image[(y - 1) * IMG_WIDTH + (x - 1)];
    int t  = image[(y - 1) * IMG_WIDTH + x];
    int tr = image[(y - 1) * IMG_WIDTH + (x + 1)];

    int l  = image[y * IMG_WIDTH + (x - 1)];
    int r  = image[y * IMG_WIDTH + (x + 1)];

    int bl = image[(y + 1) * IMG_WIDTH + (x - 1)];
    int b  = image[(y + 1) * IMG_WIDTH + x];
    int br = image[(y + 1) * IMG_WIDTH + (x + 1)];

    *gx =
        -tl + tr +
        -2 * l + 2 * r +
        -bl + br;

    *gy =
        -tl - 2 * t - tr +
         bl + 2 * b + br;
}

static int64_t harris_response_at( const uint8_t *image, int x, int y)
{
    int64_t sum_xx = 0;
    int64_t sum_yy = 0;
    int64_t sum_xy = 0;

    for (int wy = -HARRIS_WINDOW_RADIUS; wy <= HARRIS_WINDOW_RADIUS; wy++) {

        for (int wx = -HARRIS_WINDOW_RADIUS; wx <= HARRIS_WINDOW_RADIUS; wx++) {

            int gx;
            int gy;

            harris_sobel_at(
                image,
                x + wx,
                y + wy,
                &gx,
                &gy
            );

            sum_xx += (int64_t)gx * gx;
            sum_yy += (int64_t)gy * gy;
            sum_xy += (int64_t)gx * gy;
        }
    }

    int64_t determinant =
        sum_xx * sum_yy -
        sum_xy * sum_xy;

    int64_t trace =
        sum_xx + sum_yy;

    /*
     * Harris:
     *
     * R = det(M) - k * trace(M)^2
     *
     * k = 0.04 = 1/25
     */
    int64_t response =
        determinant -
        ((trace * trace) / HARRIS_K_DIVISOR);

    return response;
}

static void compute_harris_response(const uint8_t *image, uint8_t *response)
{
    memset(response, 0, IMG_PIXELS);

    int64_t max_response = 0;

    /*
     * First pass:
     * find maximum positive Harris response
     * across the WHOLE image.
     */
    for (int y = 2; y < IMG_HEIGHT - 2; y++) {
        for (int x = 2; x < IMG_WIDTH - 2; x++) {

            int64_t value =
                harris_response_at(
                    image,
                    x,
                    y
                );

            if (value > max_response) {
                max_response = value;
            }
        }
    }

    if (max_response <= 0) {
        return;
    }

    /*
     * Second pass:
     * normalize positive Harris responses
     * to 0...255.
     */
    for (int y = 2; y < IMG_HEIGHT - 2; y++) {
        for (int x = 2; x < IMG_WIDTH - 2; x++) {

            int64_t value =
                harris_response_at(
                    image,
                    x,
                    y
                );

            if (value <= 0) {
                continue;
            }

            int normalized =
                (int)(
                    (value * 255LL) /
                    max_response
                );

            if (normalized > 255) {
                normalized = 255;
            }

            response[
                y * IMG_WIDTH + x
            ] = (uint8_t)normalized;
        }
    }
}

static int find_chess_corners(const uint8_t *response)
{
    int count = 0;

    for (int y = CHESS_CORNER_NMS_RADIUS;
        y < IMG_HEIGHT - CHESS_CORNER_NMS_RADIUS;
        y++) {

        for (int x = CHESS_CORNER_NMS_RADIUS;
            x < IMG_WIDTH - CHESS_CORNER_NMS_RADIUS;
            x++) {

            int index =
                y * IMG_WIDTH + x;

            uint8_t score =
                response[index];

            if (score <
                HARRIS_RESPONSE_THRESHOLD) {
                continue;
            }

            int is_maximum = 1;

            for (int dy = -CHESS_CORNER_NMS_RADIUS;
                dy <= CHESS_CORNER_NMS_RADIUS &&
                is_maximum;
                dy++) {

                for (int dx = -CHESS_CORNER_NMS_RADIUS;
                    dx <= CHESS_CORNER_NMS_RADIUS;
                    dx++) {

                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    int neighbour =
                        (y + dy) * IMG_WIDTH +
                        (x + dx);

                    if (response[neighbour] >
                        score) {

                        is_maximum = 0;
                        break;
                    }
                }
            }

            if (!is_maximum) {
                continue;
            }

            if (count >= CHESS_CORNER_MAX) {
                return count;
            }

            chess_corners[count].x = x;
            chess_corners[count].y = y;
            chess_corners[count].score = score;

            count++;
        }
    }

    return count;
}

static void render_chess_corners(int count)
{
    memcpy(line_buffer, gray_buffer, IMG_PIXELS);

    for (int i = 0; i < count; i++) {
        int cx = chess_corners[i].x;
        int cy = chess_corners[i].y;

        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {

                int x = cx + dx;
                int y = cy + dy;

                if (x >= 0 && x < IMG_WIDTH &&
                    y >= 0 && y < IMG_HEIGHT) {
                    line_buffer[y * IMG_WIDTH + x] = 255;
                }
            }
        }
    }
}

static void grid_debug_pixel(
    uint8_t *image,
    int x,
    int y)
{
    if (x < 0 || x >= IMG_WIDTH ||
        y < 0 || y >= IMG_HEIGHT) {
        return;
    }

    int index = y * IMG_WIDTH + x;

    /*
     * Draw with opposite intensity from
     * the underlying grayscale pixel.
     */
    image[index] =
        (gray_buffer[index] > 127)
            ? 0
            : 255;
}


static void draw_grid_debug_line(
    uint8_t *image,
    const grid_line_t *line)
{
    if (!line->valid) {
        return;
    }

    /*
     * Line equation:
     *
     * a*x + b*y + c = 0
     */

    if (fabsf(line->b) >
        fabsf(line->a)) {

        /*
         * Solve for y.
         */
        for (int x = 0;
             x < IMG_WIDTH;
             x++) {

            float yf =
                -(line->a * (float)x +
                  line->c) /
                line->b;

            int y =
                (int)lroundf(yf);

            grid_debug_pixel(
                image,
                x,
                y
            );
        }

    } else {

        /*
         * Solve for x.
         */
        for (int y = 0;
             y < IMG_HEIGHT;
             y++) {

            float xf =
                -(line->b * (float)y +
                  line->c) /
                line->a;

            int x =
                (int)lroundf(xf);

            grid_debug_pixel(
                image,
                x,
                y
            );
        }
    }
}


static void draw_grid_debug_corner(
    uint8_t *image,
    const point_f_t *point)
{
    int cx =
        (int)lroundf(point->x);

    int cy =
        (int)lroundf(point->y);

    /*
     * Draw a large cross.
     */
    for (int d = -4; d <= 4; d++) {

        grid_debug_pixel(
            image,
            cx + d,
            cy
        );

        grid_debug_pixel(
            image,
            cx,
            cy + d
        );
    }
}


static void draw_grid_segment(
    uint8_t *image,
    point_f_t p0,
    point_f_t p1)
{
    int x0 = (int)lroundf(p0.x);
    int y0 = (int)lroundf(p0.y);

    int x1 = (int)lroundf(p1.x);
    int y1 = (int)lroundf(p1.y);

    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;

    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;

    int error = dx + dy;

    while (1) {

        /*
         * Make board outline thicker
         * than the internal grid.
         */
        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {

                grid_debug_pixel(
                    image,
                    x0 + ox,
                    y0 + oy
                );
            }
        }

        if (x0 == x1 &&
            y0 == y1) {
            break;
        }

        int e2 = 2 * error;

        if (e2 >= dy) {
            error += dy;
            x0 += sx;
        }

        if (e2 <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}


static void render_chess_grid_debug(const chess_grid_t *grid, uint8_t *output)
{
    /*
     * Start with original grayscale.
     */
    memcpy(
        output,
        gray_buffer,
        IMG_PIXELS
    );

    if (!grid->valid) {
        return;
    }

    /*
     * Draw all 9 lines from one family.
     */
    for (int i = 0; i < GRID_LINE_COUNT; i++) {

        draw_grid_debug_line(
            output,
            &grid->rows[i]
        );
    }

    /*
     * Draw all 9 lines from the
     * second family.
     */
    for (int i = 0; i < GRID_LINE_COUNT; i++) {

        draw_grid_debug_line(
            output,
            &grid->cols[i]
        );
    }

    /*
     * Draw the four extreme
     * intersections.
     */
    for (int i = 0; i < 4; i++) {

        draw_grid_debug_corner(
            output,
            &grid->corners[i]
        );
    }

    /*
     * Draw detected board perimeter.
     *
     * We are using the current corner
     * ordering here only for visualization.
     * We will normalize TL/TR/BR/BL next.
     */
    draw_grid_segment(
        output,
        grid->corners[0],
        grid->corners[1]
    );

    draw_grid_segment(
        output,
        grid->corners[1],
        grid->corners[2]
    );

    draw_grid_segment(
        output,
        grid->corners[2],
        grid->corners[3]
    );

    draw_grid_segment(
        output,
        grid->corners[3],
        grid->corners[0]
    );
}

static void sort_grid_centers(float *values)
{
    for (int i = 0; i < GRID_LINE_COUNT - 1; i++) {
        for (int j = i + 1; j < GRID_LINE_COUNT; j++) {
            if (values[j] < values[i]) {
                float temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
    }
}
static int cluster_grid_axis(const float *values, int count, float *centers, int *labels)
{
    uint32_t histogram[GRID_PROJECTION_HIST_BINS];
    uint32_t smooth[GRID_PROJECTION_HIST_BINS];

    memset(histogram, 0, sizeof(histogram));
    memset(smooth, 0, sizeof(smooth));

    if (count <= 0) {
        return 0;
    }

    float min_value = values[0];
    float max_value = values[0];

    for (int i = 1; i < count; i++) {

        if (values[i] < min_value) {
            min_value = values[i];
        }

        if (values[i] > max_value) {
            max_value = values[i];
        }
    }

    /*
     * Add a little margin around the
     * projection range.
     */
    float origin =
        floorf(min_value) - 2.0f;

    float end =
        ceilf(max_value) + 2.0f;

    int bin_count =
        (int)(end - origin) + 1;

    if (bin_count <= 0 ||
        bin_count > GRID_PROJECTION_HIST_BINS) {

        return 0;
    }

    /*
     * Build weighted histogram.
     *
     * Each Harris point votes at its
     * projected location.
     *
     * 256 provides a base vote so even
     * weaker Harris points still matter.
     */
    for (int i = 0; i < count; i++) {

        int bin =
            (int)lroundf(
                values[i] - origin
            );

        if (bin < 0 ||
            bin >= bin_count) {
            continue;
        }

        uint32_t weight =
            256U +
            (uint32_t)chess_corners[i].score;

        histogram[bin] += weight;
    }

    /*
     * Smooth using:
     *
     * 1 2 3 2 1
     */
    for (int b = 0; b < bin_count; b++) {

        uint32_t value =
            3U * histogram[b];

        if (b >= 1) {
            value +=
                2U * histogram[b - 1];
        }

        if (b + 1 < bin_count) {
            value +=
                2U * histogram[b + 1];
        }

        if (b >= 2) {
            value +=
                histogram[b - 2];
        }

        if (b + 2 < bin_count) {
            value +=
                histogram[b + 2];
        }

        smooth[b] = value;
    }

    int selected[GRID_LINE_COUNT];

    for (int i = 0; i < GRID_LINE_COUNT; i++) {

        selected[i] = -1;
    }

    /*
     * Select the nine strongest separated
     * peaks.
     */
    for (int peak = 0; peak < GRID_LINE_COUNT; peak++) {

        int best_bin = -1;
        uint32_t best_score = 0;

        for (int b = 0; b < bin_count; b++) {

            if (smooth[b] == 0) {
                continue;
            }

            /*
             * Must be a local maximum.
             */
            if (b > 0 &&
                smooth[b] < smooth[b - 1]) {
                continue;
            }

            if (b + 1 < bin_count &&
                smooth[b] < smooth[b + 1]) {
                continue;
            }

            /*
             * Do not select another peak
             * too close to one already used.
             */
            int too_close = 0;

            for (int j = 0; j < peak; j++) {

                if (abs(
                        b -
                        selected[j]
                    ) <
                    GRID_PEAK_MIN_SEPARATION) {

                    too_close = 1;
                    break;
                }
            }

            if (too_close) {
                continue;
            }

            if (smooth[b] > best_score) {

                best_score =
                    smooth[b];

                best_bin = b;
            }
        }

        if (best_bin < 0) {
            return 0;
        }

        selected[peak] =
            best_bin;
    }

    /*
     * Sort peaks from low projection
     * to high projection.
     */
    for (int i = 0; i < GRID_LINE_COUNT - 1; i++) {

        for (int j = i + 1; j < GRID_LINE_COUNT; j++) {

            if (selected[j] <
                selected[i]) {

                int temporary =
                    selected[i];

                selected[i] =
                    selected[j];

                selected[j] =
                    temporary;
            }
        }
    }

    /*
     * Convert histogram positions back
     * into projection coordinates.
     */
    for (int i = 0; i < GRID_LINE_COUNT; i++) {

        centers[i] =
            origin +
            (float)selected[i];
    }

    /*
     * Sanity check spacing.
     *
     * Your current board has roughly
     * 10-15 px between neighbouring
     * projected grid lines.
     */
    for (int i = 1; i < GRID_LINE_COUNT; i++) {

        float spacing =
            centers[i] -
            centers[i - 1];

        if (spacing < 5.0f ||
            spacing > 22.0f) {

            return 0;
        }
    }

    int cluster_counts[GRID_LINE_COUNT];

    memset(
        cluster_counts,
        0,
        sizeof(cluster_counts)
    );

    /*
     * Assign each Harris candidate to
     * its nearest detected grid band.
     *
     * Far-away points are rejected as
     * outliers instead of contaminating
     * the nearest line.
     */
    for (int i = 0; i < count; i++) {

        int best_cluster = -1;
        float best_distance = 1000000.0f;

        for (int c = 0; c < GRID_LINE_COUNT; c++) {

            float distance =
                fabsf(
                    values[i] -
                    centers[c]
                );

            if (distance <
                best_distance) {

                best_distance =
                    distance;

                best_cluster = c;
            }
        }

        if (best_cluster >= 0 &&
            best_distance <=
                GRID_ASSIGN_MAX_DISTANCE) {

            labels[i] =
                best_cluster;

            cluster_counts[
                best_cluster]++;

        } else {

            labels[i] = -1;
        }
    }

    /*
     * Every detected chessboard line
     * must have at least two supporting
     * Harris points.
     */
    for (int c = 0; c < GRID_LINE_COUNT; c++) {

        if (cluster_counts[c] <
            GRID_MIN_POINTS_PER_LINE) {

            return 0;
        }
    }

    return 1;
}

static float grid_line_angle_difference(float a, float b)
{
    float difference = fabsf(a - b);

    while (difference >= 180.0f) {
        difference -= 180.0f;
    }

    if (difference > 90.0f) {
        difference = 180.0f - difference;
    }

    return difference;
}


static grid_line_t fit_grid_line(int cluster, const int *labels,  int point_count, float expected_angle_deg)
{
    grid_line_t line = {0};

    float mean_x = 0.0f;
    float mean_y = 0.0f;

    int count = 0;

    /*
     * Calculate cluster centroid.
     */
    for (int i = 0; i < point_count; i++) {

        if (labels[i] != cluster) {
            continue;
        }

        mean_x += chess_corners[i].x;
        mean_y += chess_corners[i].y;

        count++;
    }

    line.point_count = count;

    if (count < GRID_MIN_POINTS_PER_LINE) {
        return line;
    }

    mean_x /= (float)count;
    mean_y /= (float)count;

    /*
     * Calculate local PCA direction.
     */
    float xx = 0.0f;
    float yy = 0.0f;
    float xy = 0.0f;

    for (int i = 0; i < point_count; i++) {

        if (labels[i] != cluster) {
            continue;
        }

        float dx =
            chess_corners[i].x -
            mean_x;

        float dy =
            chess_corners[i].y -
            mean_y;

        xx += dx * dx;
        yy += dy * dy;
        xy += dx * dy;
    }

    float local_angle =
        0.5f *
        atan2f(
            2.0f * xy,
            xx - yy
        );

    float local_angle_deg =
        local_angle *
        180.0f /
        PI_F;

    if (local_angle_deg < 0.0f) {
        local_angle_deg += 180.0f;
    }

    /*
     * Start with the globally detected
     * grid direction.
     */
    float chosen_angle_deg =
        expected_angle_deg;

    /*
     * Only trust the local PCA direction
     * when:
     *
     * 1. We have at least 4 points.
     * 2. It agrees with the global grid
     *    direction within 12 degrees.
     *
     * Otherwise use the robust global
     * grid direction.
     */
    if (count >= 4) {

        float angle_error =
            grid_line_angle_difference(
                local_angle_deg,
                expected_angle_deg
            );

        if (angle_error <=
            GRID_MAX_LINE_ANGLE_ERROR_DEG) {

            chosen_angle_deg =
                local_angle_deg;
        }
    }

    float chosen_angle =
        chosen_angle_deg *
        PI_F /
        180.0f;

    float dx =
        cosf(chosen_angle);

    float dy =
        sinf(chosen_angle);

    /*
     * Normal vector.
     */
    line.a = -dy;
    line.b = dx;

    /*
     * Force the line through the
     * cluster centroid.
     */
    line.c =
        -(line.a * mean_x +
          line.b * mean_y);

    line.valid = 1;

    return line;
}

static int grid_angle_bin_distance(int a, int b)
{
    int d = abs(a - b);

    if (d > GRID_ANGLE_BINS / 2) {
        d = GRID_ANGLE_BINS - d;
    }

    return d;
}

static int detect_grid_axes(int corner_count, float *angle_a_deg, float *angle_b_deg)
{
    uint32_t histogram[GRID_ANGLE_BINS] = {0};
    uint32_t smooth[GRID_ANGLE_BINS] = {0};

    const int min_dist2 =
        GRID_PAIR_MIN_DIST *
        GRID_PAIR_MIN_DIST;

    const int max_dist2 =
        GRID_PAIR_MAX_DIST *
        GRID_PAIR_MAX_DIST;

    for (int i = 0; i < corner_count; i++) {

        for (int j = i + 1; j < corner_count; j++) {

            int dx =
                chess_corners[j].x -
                chess_corners[i].x;

            int dy =
                chess_corners[j].y -
                chess_corners[i].y;

            int dist2 =
                dx * dx +
                dy * dy;

            if (dist2 < min_dist2 ||
                dist2 > max_dist2) {
                continue;
            }

            float angle =
                atan2f(
                    (float)dy,
                    (float)dx
                ) *
                180.0f /
                PI_F;

            if (angle < 0.0f) {
                angle += 180.0f;
            }

            if (angle >= 180.0f) {
                angle -= 180.0f;
            }

            int bin =
                (int)((angle + 1.0f) / 2.0f);

            if (bin >= GRID_ANGLE_BINS) {
                bin -= GRID_ANGLE_BINS;
            }

            uint32_t strength =
                chess_corners[i].score <
                chess_corners[j].score
                    ? chess_corners[i].score
                    : chess_corners[j].score;

            /*
             * Short, strong connections receive
             * the largest vote.
             */
            uint32_t weight =
                (strength * 256U) /
                (uint32_t)dist2;

            histogram[bin] += weight;
        }
    }

    /*
     * Circular smoothing.
     */
    for (int i = 0; i < GRID_ANGLE_BINS; i++) {

        int previous =
            (i + GRID_ANGLE_BINS - 1) %
            GRID_ANGLE_BINS;

        int next =
            (i + 1) %
            GRID_ANGLE_BINS;

        smooth[i] =
            histogram[previous] +
            2U * histogram[i] +
            histogram[next];
    }

    uint64_t best_score = 0;
    int best_a = -1;
    int best_b = -1;

    /*
     * Find two approximately perpendicular
     * dominant directions.
     *
     * 30 bins = 60 degrees.
     * 45 bins = 90 degrees.
     */
    for (int a = 0; a < GRID_ANGLE_BINS; a++) {

        for (int b = a + 1; b < GRID_ANGLE_BINS; b++) {

            int separation =
                grid_angle_bin_distance(
                    a,
                    b
                );

            if (separation <
                GRID_AXIS_MIN_SEP_BINS) {
                continue;
            }

            uint64_t score =
                (uint64_t)smooth[a] +
                (uint64_t)smooth[b];

            if (score > best_score) {
                best_score = score;
                best_a = a;
                best_b = b;
            }
        }
    }

    if (best_a < 0 ||
        best_b < 0 ||
        best_score == 0) {

        return 0;
    }

    *angle_a_deg =
        best_a * 2.0f;

    *angle_b_deg =
        best_b * 2.0f;

    return 1;
}

static int intersect_grid_lines(const grid_line_t *line1, const grid_line_t *line2, point_f_t *point)
{
    float determinant =
        line1->a * line2->b -
        line2->a * line1->b;

    /*
     * Parallel or nearly parallel lines
     * do not have a reliable intersection.
     */
    if (fabsf(determinant) < 0.0001f) {
        return 0;
    }

    point->x =
        (line1->b * line2->c -
         line2->b * line1->c) /
        determinant;

    point->y =
        (line2->a * line1->c -
         line1->a * line2->c) /
        determinant;

    return 1;
}

static void order_chess_grid_corners(chess_grid_t *grid)
{
    point_f_t original[4];

    for (int i = 0; i < 4; i++) {
        original[i] = grid->corners[i];
    }

    int tl_index = 0;
    int tr_index = 0;
    int br_index = 0;
    int bl_index = 0;

    float min_sum =
        original[0].x + original[0].y;

    float max_sum =
        min_sum;

    float min_diff =
        original[0].x - original[0].y;

    float max_diff =
        min_diff;

    for (int i = 1; i < 4; i++) {

        float sum =
            original[i].x +
            original[i].y;

        float diff =
            original[i].x -
            original[i].y;

        /*
         * x + y:
         *
         * smallest -> top-left
         * largest  -> bottom-right
         */
        if (sum < min_sum) {
            min_sum = sum;
            tl_index = i;
        }

        if (sum > max_sum) {
            max_sum = sum;
            br_index = i;
        }

        /*
         * x - y:
         *
         * largest  -> top-right
         * smallest -> bottom-left
         */
        if (diff > max_diff) {
            max_diff = diff;
            tr_index = i;
        }

        if (diff < min_diff) {
            min_diff = diff;
            bl_index = i;
        }
    }

    grid->corners[0] =
        original[tl_index];

    grid->corners[1] =
        original[tr_index];

    grid->corners[2] =
        original[br_index];

    grid->corners[3] =
        original[bl_index];
}

static int detect_chess_grid(int corner_count, chess_grid_t *grid)
{
    memset(
        grid,
        0,
        sizeof(*grid)
    );

    if (corner_count < 30) {
        return 0;
    }

    static float projection_a[CHESS_CORNER_MAX];
    static float projection_b[CHESS_CORNER_MAX];

    static int labels_a[CHESS_CORNER_MAX];
    static int labels_b[CHESS_CORNER_MAX];

    float angle_a_deg;
    float angle_b_deg;

    if (!detect_grid_axes(corner_count, &angle_a_deg, &angle_b_deg)) {

        return 0;
    }

    ESP_LOGI(TAG,
        "GRID AXES A=%.1f B=%.1f",
        angle_a_deg,
        angle_b_deg
    );

    float angle_a =
        angle_a_deg *
        PI_F /
        180.0f;

    float angle_b =
        angle_b_deg *
        PI_F /
        180.0f;

    /*
     * Normals to the two detected
     * chessboard directions.
     */
    float normal_a_x =
        -sinf(angle_a);

    float normal_a_y =
        cosf(angle_a);

    float normal_b_x =
        -sinf(angle_b);

    float normal_b_y =
        cosf(angle_b);

    /*
     * Project every Harris point onto
     * the normals of the two grid
     * directions.
     */
    for (int i = 0; i < corner_count; i++) {

        float x =
            chess_corners[i].x;

        float y =
            chess_corners[i].y;

        projection_a[i] =
            x * normal_a_x +
            y * normal_a_y;

        projection_b[i] =
            x * normal_b_x +
            y * normal_b_y;
    }

    float centers_a[GRID_LINE_COUNT];
    float centers_b[GRID_LINE_COUNT];

    if (!cluster_grid_axis(projection_a, corner_count, centers_a, labels_a)) {

    ESP_LOGW(
        TAG,
        "GRID FAIL axis A clustering"
    );

    return 0;
}

if (!cluster_grid_axis(projection_b, corner_count, centers_b, labels_b)) {

    ESP_LOGW(
        TAG,
        "GRID FAIL axis B clustering"
    );

    ESP_LOGI(
    TAG,
    "GRID BANDS "
    "A=[%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f] "
    "B=[%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f]",
    centers_a[0],
    centers_a[1],
    centers_a[2],
    centers_a[3],
    centers_a[4],
    centers_a[5],
    centers_a[6],
    centers_a[7],
    centers_a[8],
    centers_b[0],
    centers_b[1],
    centers_b[2],
    centers_b[3],
    centers_b[4],
    centers_b[5],
    centers_b[6],
    centers_b[7],
    centers_b[8]
);

    return 0;
}

    /*
     * Points having similar projection A
     * belong to lines parallel to axis A.
     */
    for (int i = 0; i < GRID_LINE_COUNT; i++) {

        grid->rows[i] =
            fit_grid_line(
                i,
                labels_a,
                corner_count,
                angle_a_deg
            );

        grid->cols[i] =
            fit_grid_line(
                i,
                labels_b,
                corner_count,
                angle_b_deg
            );
    }

    /*
     * Make sure every one of the
     * 7 + 7 fitted lines is valid.
     */
    for (int i = 0; i < GRID_LINE_COUNT; i++) {

        if (!grid->rows[i].valid ||
            !grid->cols[i].valid) {

            return 0;
        }
    }

    /*
     * Internal 7x7 lattice corners.
     *
     * These are NOT yet the physical
     * outer corners of the 8x8 board.
     */
    if (!intersect_grid_lines(
            &grid->rows[0],
            &grid->cols[0],
            &grid->corners[0])) {
        return 0;
    }

    if (!intersect_grid_lines(
            &grid->rows[0],
            &grid->cols[
                GRID_LINE_COUNT - 1],
            &grid->corners[1])) {
        return 0;
    }

    if (!intersect_grid_lines(
            &grid->rows[
                GRID_LINE_COUNT - 1],
            &grid->cols[
                GRID_LINE_COUNT - 1],
            &grid->corners[2])) {
        return 0;
    }

    if (!intersect_grid_lines(
            &grid->rows[
                GRID_LINE_COUNT - 1],
            &grid->cols[0],
            &grid->corners[3])) {
        return 0;
    }

    /*
     * Basic sanity check.
     */
    for (int i = 0; i < 4; i++) {

        if (grid->corners[i].x < -10.0f ||
            grid->corners[i].x >
                IMG_WIDTH + 10.0f ||
            grid->corners[i].y < -10.0f ||
            grid->corners[i].y >
                IMG_HEIGHT + 10.0f) {

            return 0;
        }
    }

    order_chess_grid_corners(grid);

    grid->valid = 1;

    return 1;
}

static int compute_board_projective_map(const chess_grid_t *grid, projective_map_t *map)
{
    float x0 = grid->corners[0].x;
    float y0 = grid->corners[0].y;

    float x1 = grid->corners[1].x;
    float y1 = grid->corners[1].y;

    float x2 = grid->corners[2].x;
    float y2 = grid->corners[2].y;

    float x3 = grid->corners[3].x;
    float y3 = grid->corners[3].y;

    float dx1 = x1 - x2;
    float dx2 = x3 - x2;
    float dx3 = x0 - x1 + x2 - x3;

    float dy1 = y1 - y2;
    float dy2 = y3 - y2;
    float dy3 = y0 - y1 + y2 - y3;

    float denominator =
        dx1 * dy2 -
        dx2 * dy1;

    if (fabsf(denominator) < 0.0001f) {
        return 0;
    }

    map->g =
        (dx3 * dy2 -
         dx2 * dy3) /
        denominator;

    map->h =
        (dx1 * dy3 -
         dx3 * dy1) /
        denominator;

    map->a =
        x1 - x0 +
        map->g * x1;

    map->b =
        x3 - x0 +
        map->h * x3;

    map->c = x0;

    map->d =
        y1 - y0 +
        map->g * y1;

    map->e =
        y3 - y0 +
        map->h * y3;

    map->f = y0;

    return 1;
}


static int warp_chessboard(const uint8_t *source, uint8_t *destination, const chess_grid_t *grid)
{
    if (!grid->valid) {
        return 0;
    }

    projective_map_t map;

    if (!compute_board_projective_map(
            grid,
            &map)) {
        return 0;
    }

    for (int y = 0; y < BOARD_WARP_SIZE; y++) {

        float v =
            (float)y /
            (float)(BOARD_WARP_SIZE - 1);

        for (int x = 0; x < BOARD_WARP_SIZE; x++) {

            float u =
                (float)x /
                (float)(BOARD_WARP_SIZE - 1);

            float denominator =
                map.g * u +
                map.h * v +
                1.0f;

            if (fabsf(denominator) < 0.0001f) {

                destination[
                    y * BOARD_WARP_SIZE + x
                ] = 0;

                continue;
            }

            float source_x = (map.a * u + map.b * v + map.c) / denominator;

            float source_y = (map.d * u + map.e * v + map.f) / denominator;

            if (source_x < 0.0f ||
                source_x > (float)(IMG_WIDTH - 1) ||
                source_y < 0.0f ||
                source_y > (float)(IMG_HEIGHT - 1)) {

                destination[
                    y * BOARD_WARP_SIZE + x
                ] = 0;

                continue;
            }

            int x0 = (int)floorf(source_x);
            int y0 = (int)floorf(source_y);

            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x1 >= IMG_WIDTH) {
                x1 = IMG_WIDTH - 1;
            }

            if (y1 >= IMG_HEIGHT) {
                y1 = IMG_HEIGHT - 1;
            }

            float fx =
                source_x - (float)x0;

            float fy =
                source_y - (float)y0;

            float p00 =
                source[y0 * IMG_WIDTH + x0];

            float p10 =
                source[y0 * IMG_WIDTH + x1];

            float p01 =
                source[y1 * IMG_WIDTH + x0];

            float p11 =
                source[y1 * IMG_WIDTH + x1];

            float top =
                p00 +
                fx * (p10 - p00);

            float bottom =
                p01 +
                fx * (p11 - p01);

            float value =
                top +
                fy * (bottom - top);

            if (value < 0.0f) {
                value = 0.0f;
            }

            if (value > 255.0f) {
                value = 255.0f;
            }

            destination[
                y * BOARD_WARP_SIZE + x
            ] =
                (uint8_t)(value + 0.5f);
        }
    }

    return 1;
}

/* BENCHMARK */

static esp_err_t run_benchmark(const char *run_directory)
{
    /* ============================================================
     * CSV FILE
     * ============================================================ */

    char csv_path[128];

    snprintf(
        csv_path,
        sizeof(csv_path),
        "%s/bench.csv",
        run_directory
    );

    FILE *csv = fopen(csv_path, "w");

    if (csv == NULL) {
        ESP_LOGE(TAG, "Failed to open benchmark CSV");
        return ESP_FAIL;
    }


    /*
     * CSV columns.
     *
     * The old Sobel/Hough pipeline is retained for comparison.
     * The final selected chessboard-localization pipeline is:
     *
     * capture
     *   -> Y extraction
     *   -> Gaussian
     *   -> Harris/grid
     *   -> perspective warp
     */
    fprintf(
        csv,

        "frame,width,height,frame_bytes,"
        "capture_us,"
        "extract_y_us,"
        "gaussian_us,"
        "sobel_us,"
        "threshold_us,"
        "dsp_total_us,"
        "pipeline_us,"
        "hough_vote_us,"
        "hough_peak_us,"
        "hough_total_us,"
        "phase3a_algorithm_us,"
        "phase3a_pipeline_us,"
        "line_count,"
        "family_detect_us,"
        "phase3b_algorithm_us,"
        "phase3b_pipeline_us,"
        "family_a_theta_deg,"
        "family_b_theta_deg,"
        "family_a_count,"
        "family_b_count,"
        "free_internal_ram,"
        "free_psram,"
        "chess_grid_us,"
        "warp_us,"
        "localization_algorithm_us,"
        "localization_pipeline_us,"
        "grid_found,"
        "warp_ok\n"
    );

    fflush(csv);


    /* ============================================================
     * CAMERA WARM-UP
     * ============================================================ */

    ESP_LOGI(
        TAG,
        "Camera warm-up: %d frames",
        WARMUP_FRAMES
    );

    for (int i = 0; i < WARMUP_FRAMES; i++) {

        camera_fb_t *fb =
            esp_camera_fb_get();

        if (fb != NULL) {
            esp_camera_fb_return(fb);
        }

        vTaskDelay(
            pdMS_TO_TICKS(50)
        );
    }


    /* ============================================================
     * ACCUMULATORS
     * ============================================================ */

    uint64_t capture_total = 0;
    uint64_t extract_total = 0;
    uint64_t gaussian_total = 0;
    uint64_t sobel_total = 0;
    uint64_t threshold_total = 0;

    uint64_t dsp_total = 0;
    uint64_t pipeline_total = 0;

    uint64_t chess_grid_total = 0;
    uint64_t warp_total = 0;

    uint64_t localization_algorithm_total = 0;
    uint64_t localization_pipeline_total = 0;


    /* ------------------------------------------------------------
     * Min/max timing
     * ------------------------------------------------------------ */

    int64_t capture_min = INT64_MAX;
    int64_t capture_max = 0;

    int64_t chess_grid_min = INT64_MAX;
    int64_t chess_grid_max = 0;

    int64_t warp_min = INT64_MAX;
    int64_t warp_max = 0;

    int64_t localization_min = INT64_MAX;
    int64_t localization_max = 0;


    /* ------------------------------------------------------------
     * Counters
     * ------------------------------------------------------------ */

    int successful_frames = 0;
    int saved_images = 0;

    int grid_success_count = 0;
    int warp_success_count = 0;


    /* ------------------------------------------------------------
     * Reset Hough-family tracking
     * ------------------------------------------------------------ */

    tracked_family_a_theta = -1;
    tracked_family_b_theta = -1;
    family_track_misses = 0;


    /* ============================================================
     * BENCHMARK LOOP
     * ============================================================ */

    for (int frame = 1; frame <= TEST_FRAMES; frame++) {


        /* ========================================================
         * CAMERA CAPTURE
         * ======================================================== */

        int64_t capture_start =
            esp_timer_get_time();

        camera_fb_t *fb =
            esp_camera_fb_get();

        int64_t capture_end =
            esp_timer_get_time();


        if (fb == NULL) {

            ESP_LOGE(
                TAG,
                "Camera capture failed frame=%d",
                frame
            );

            /*
             * bench.csv contains 32 columns.
             *
             * Write:
             * frame,width,height,bytes,capture
             *
             * then fill the remaining 27 columns with zero.
             */
            fprintf(
                csv,
                "%d,0,0,0,-1",
                frame
            );

            for (int column = 5; column < 32; column++) {

                fprintf(csv, ",0");
            }

            fprintf(csv, "\n");

            fflush(csv);

            continue;
        }


        unsigned int frame_width =
            fb->width;

        unsigned int frame_height =
            fb->height;

        unsigned int frame_bytes =
            fb->len;


        int64_t capture_us =
            capture_end -
            capture_start;


        if (capture_us < capture_min) {
            capture_min = capture_us;
        }

        if (capture_us > capture_max) {
            capture_max = capture_us;
        }


        /* --------------------------------------------------------
         * Verify expected YUV422 QQVGA frame.
         * -------------------------------------------------------- */

        if (frame_width != IMG_WIDTH ||
            frame_height != IMG_HEIGHT ||
            frame_bytes < IMG_PIXELS * 2) {

            ESP_LOGE(
                TAG,
                "Unexpected frame: %ux%u, %u bytes",
                frame_width,
                frame_height,
                frame_bytes
            );


            /*
             * Preserve the failed frame in the CSV.
             */
            fprintf(
                csv,
                "%d,%u,%u,%u,%lld",
                frame,
                frame_width,
                frame_height,
                frame_bytes,
                (long long)capture_us
            );

            for (int column = 5; column < 32; column++) {

                fprintf(csv, ",0");
            }

            fprintf(csv, "\n");

            fflush(csv);

            esp_camera_fb_return(fb);

            continue;
        }


        /* ========================================================
         * Y-CHANNEL EXTRACTION
         * ======================================================== */

        int64_t t_extract_start =
            esp_timer_get_time();

        extract_y_channel(
            fb,
            gray_buffer
        );

        int64_t t_extract_end =
            esp_timer_get_time();


        /*
         * Return camera framebuffer as early as possible.
         *
         * This allows the camera driver to continue acquiring
         * the next frame while DSP processing is performed.
         */
        esp_camera_fb_return(fb);
        fb = NULL;


        /* ========================================================
         * GAUSSIAN FILTER
         * ======================================================== */

        int64_t t_gaussian_start =
            esp_timer_get_time();

        gaussian_3x3(
            gray_buffer,
            blur_buffer
        );

        int64_t t_gaussian_end =
            esp_timer_get_time();


        /* ========================================================
         * LEGACY SOBEL / EDGE PIPELINE
         *
         * Retained for algorithm comparison and debug images.
         * It is NOT part of final Harris-based localization timing.
         * ======================================================== */

        int64_t t_sobel_start =
            esp_timer_get_time();

        sobel_filter(
            blur_buffer,
            gradient_buffer
        );

        int64_t t_sobel_end =
            esp_timer_get_time();


        int64_t t_threshold_start =
            esp_timer_get_time();

        threshold_edges(
            gradient_buffer,
            edge_buffer
        );

        int64_t t_threshold_end =
            esp_timer_get_time();


        /* ========================================================
         * LEGACY ROI DETECTION
         * ======================================================== */

        int64_t t_roi_start =
            esp_timer_get_time();

        edge_roi_t board_roi;

        detect_edge_roi(
            edge_buffer,
            &board_roi
        );

        int64_t t_roi_end =
            esp_timer_get_time();


        /* ========================================================
         * FINAL HARRIS / GRID LOCALIZATION
         * ======================================================== */

        int64_t t_chess_corner_start =
            esp_timer_get_time();


        /*
         * Compute Harris response from Gaussian-smoothed image.
         */
        compute_harris_response(
            blur_buffer,
            line_buffer
        );


        /*
         * Non-maximum suppression and candidate extraction.
         */
        int chess_corner_count =
            find_chess_corners(
                line_buffer
            );


        /*
         * Recover the complete 9x9 chessboard lattice.
         */
        int grid_found =
            detect_chess_grid(
                chess_corner_count,
                &detected_grid
            );


        int64_t t_chess_corner_end =
            esp_timer_get_time();


        int64_t chess_corner_us =
            t_chess_corner_end -
            t_chess_corner_start;


        /*
         * IMPORTANT:
         *
         * Logging is outside the timer so UART output does not
         * contaminate Harris/grid measurements.
         */
        ESP_LOGI(
            TAG,
            "HARRIS frame=%d corners=%d time=%.3f ms",
            frame,
            chess_corner_count,
            chess_corner_us / 1000.0
        );


        if (grid_found) {

            ESP_LOGI(
                TAG,

                "GRID FOUND "
                "TL=(%.1f,%.1f) "
                "TR=(%.1f,%.1f) "
                "BR=(%.1f,%.1f) "
                "BL=(%.1f,%.1f)",

                detected_grid.corners[0].x,
                detected_grid.corners[0].y,

                detected_grid.corners[1].x,
                detected_grid.corners[1].y,

                detected_grid.corners[2].x,
                detected_grid.corners[2].y,

                detected_grid.corners[3].x,
                detected_grid.corners[3].y
            );

        } else {

            ESP_LOGW(
                TAG,
                "GRID NOT FOUND frame=%d",
                frame
            );
        }


        /* ========================================================
         * FINAL PERSPECTIVE CORRECTION
         * ======================================================== */

        int64_t t_warp_start =
            esp_timer_get_time();


        int warp_ok = 0;


        if (grid_found) {

            warp_ok =
                warp_chessboard(
                    gray_buffer,
                    board_warp_buffer,
                    &detected_grid
                );
        }


        int64_t t_warp_end =
            esp_timer_get_time();


        int64_t warp_us =
            t_warp_end -
            t_warp_start;


        /* ========================================================
         * LEGACY HOUGH COMPARISON PIPELINE
         * ======================================================== */

        int64_t t_hough_vote_start =
            esp_timer_get_time();


        hough_vote_masked(
            edge_buffer
        );


        int64_t t_hough_vote_end =
            esp_timer_get_time();


        /* --------------------------------------------------------
         * Detect two dominant Hough line families.
         * -------------------------------------------------------- */

        int64_t t_family_start =
            esp_timer_get_time();


        int family_a_theta = -1;
        int family_b_theta = -1;

        int family_a_count = 0;
        int family_b_count = 0;


        detect_hough_families(
            &family_a_theta,
            &family_b_theta,
            &family_a_count,
            &family_b_count
        );


        int64_t t_family_end =
            esp_timer_get_time();


        /* --------------------------------------------------------
         * Legacy board outline.
         * -------------------------------------------------------- */

        board_detection_t board = {0};


        int board_found =
            detect_board_corners(
                family_a_theta,
                family_b_theta,
                family_a_count,
                family_b_count,
                &board
            );


        if (board_found) {

            ESP_LOGI(
                TAG,

                "HOUGH BOARD "
                "frame=%d "
                "corners="
                "(%.1f,%.1f) "
                "(%.1f,%.1f) "
                "(%.1f,%.1f) "
                "(%.1f,%.1f)",

                frame,

                board.corner[0].x,
                board.corner[0].y,

                board.corner[1].x,
                board.corner[1].y,

                board.corner[2].x,
                board.corner[2].y,

                board.corner[3].x,
                board.corner[3].y
            );
        }


        /* --------------------------------------------------------
         * Hough peak extraction.
         * -------------------------------------------------------- */

        int64_t t_hough_peak_start =
            esp_timer_get_time();


        int line_count =
            find_hough_lines();


        int64_t t_hough_peak_end =
            esp_timer_get_time();


        /* ========================================================
         * TIMING CALCULATIONS
         * ======================================================== */

        int64_t extract_us =
            t_extract_end -
            t_extract_start;


        int64_t gaussian_us =
            t_gaussian_end -
            t_gaussian_start;


        int64_t sobel_us =
            t_sobel_end -
            t_sobel_start;


        int64_t threshold_us =
            t_threshold_end -
            t_threshold_start;


        int64_t roi_detect_us =
            t_roi_end -
            t_roi_start;


        /*
         * Original basic DSP pipeline.
         */
        int64_t dsp_total_us =
            extract_us +
            gaussian_us +
            sobel_us +
            threshold_us;


        int64_t pipeline_us =
            capture_us +
            dsp_total_us;


        /*
         * Hough timing.
         */
        int64_t hough_vote_us =
            t_hough_vote_end -
            t_hough_vote_start;


        int64_t hough_peak_us =
            t_hough_peak_end -
            t_hough_peak_start;


        int64_t hough_total_us =
            hough_vote_us +
            hough_peak_us;


        /*
         * Historical Phase 3A:
         *
         * Basic DSP + global Hough.
         */
        int64_t phase3a_algorithm_us =
            dsp_total_us +
            hough_total_us;


        int64_t phase3a_pipeline_us =
            capture_us +
            phase3a_algorithm_us;


        /*
         * Hough family detection.
         */
        int64_t family_detect_us =
            t_family_end -
            t_family_start;


        /*
         * Historical Phase 3B:
         *
         * DSP + ROI + Hough vote + family detection.
         */
        int64_t phase3b_algorithm_us =
            dsp_total_us +
            roi_detect_us +
            hough_vote_us +
            family_detect_us;


        int64_t phase3b_pipeline_us =
            capture_us +
            phase3b_algorithm_us;


        /*
         * --------------------------------------------------------
         * FINAL SELECTED LOCALIZATION PIPELINE
         * --------------------------------------------------------
         *
         * Only the stages required by the final Harris/grid
         * implementation are included.
         *
         * Sobel, threshold, ROI and Hough are NOT required.
         */
        int64_t localization_algorithm_us =
            extract_us +
            gaussian_us +
            chess_corner_us +
            warp_us;


        int64_t localization_pipeline_us =
            capture_us +
            localization_algorithm_us;


        /* ========================================================
         * UPDATE TOTALS
         * ======================================================== */

        capture_total +=
            capture_us;

        extract_total +=
            extract_us;

        gaussian_total +=
            gaussian_us;

        sobel_total +=
            sobel_us;

        threshold_total +=
            threshold_us;

        dsp_total +=
            dsp_total_us;

        pipeline_total +=
            pipeline_us;


        chess_grid_total +=
            chess_corner_us;


        localization_algorithm_total +=
            localization_algorithm_us;


        localization_pipeline_total +=
            localization_pipeline_us;


        /* --------------------------------------------------------
         * Harris/grid min/max.
         * -------------------------------------------------------- */

        if (chess_corner_us < chess_grid_min) {
            chess_grid_min =
                chess_corner_us;
        }

        if (chess_corner_us > chess_grid_max) {
            chess_grid_max =
                chess_corner_us;
        }


        /* --------------------------------------------------------
         * Final localization min/max.
         * -------------------------------------------------------- */

        if (localization_algorithm_us <
            localization_min) {

            localization_min =
                localization_algorithm_us;
        }


        if (localization_algorithm_us >
            localization_max) {

            localization_max =
                localization_algorithm_us;
        }


        /* --------------------------------------------------------
         * Detection success counts.
         * -------------------------------------------------------- */

        if (grid_found) {

            grid_success_count++;
        }


        if (warp_ok) {

            warp_success_count++;

            warp_total +=
                warp_us;


            if (warp_us < warp_min) {

                warp_min =
                    warp_us;
            }


            if (warp_us > warp_max) {

                warp_max =
                    warp_us;
            }
        }


        successful_frames++;


        /* ========================================================
         * MEMORY MEASUREMENT
         * ======================================================== */

        size_t free_internal =
            heap_caps_get_free_size(
                MALLOC_CAP_INTERNAL |
                MALLOC_CAP_8BIT
            );


        size_t free_psram =
            heap_caps_get_free_size(
                MALLOC_CAP_SPIRAM
            );


        /* ========================================================
         * WRITE CSV ROW
         * ======================================================== */

        fprintf(
            csv,

            "%d,%u,%u,%u,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%d,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%d,"
            "%d,"
            "%d,"
            "%d,"
            "%u,"
            "%u,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%lld,"
            "%d,"
            "%d\n",

            frame,

            frame_width,
            frame_height,
            frame_bytes,

            (long long)capture_us,

            (long long)extract_us,

            (long long)gaussian_us,

            (long long)sobel_us,

            (long long)threshold_us,

            (long long)dsp_total_us,

            (long long)pipeline_us,

            (long long)hough_vote_us,

            (long long)hough_peak_us,

            (long long)hough_total_us,

            (long long)phase3a_algorithm_us,

            (long long)phase3a_pipeline_us,

            line_count,

            (long long)family_detect_us,

            (long long)phase3b_algorithm_us,

            (long long)phase3b_pipeline_us,

            family_a_theta >= 0
                ? family_a_theta *
                HOUGH_THETA_STEP_DEG
                : -1,

            family_b_theta >= 0
                ? family_b_theta *
                HOUGH_THETA_STEP_DEG
                : -1,

            family_a_count,
            family_b_count,

            (unsigned)free_internal,

            (unsigned)free_psram,

            (long long)chess_corner_us,

            (long long)warp_us,

            (long long)localization_algorithm_us,

            (long long)localization_pipeline_us,

            grid_found,

            warp_ok
        );


        fflush(csv);


        /* ========================================================
         * SAVE DIAGNOSTIC IMAGES
         *
         * Image saving is OUTSIDE all algorithm timers.
         * ======================================================== */

        if ((frame % SAVE_EVERY_N) == 0) {


            /* ----------------------------------------------------
             * Y = luminance
             * ---------------------------------------------------- */

            if (save_gray_pgm(
                    run_directory,
                    "Y",
                    frame,
                    gray_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * G = Gaussian
             * ---------------------------------------------------- */

            if (save_gray_pgm(
                    run_directory,
                    "G",
                    frame,
                    blur_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * M = Sobel magnitude
             * ---------------------------------------------------- */

            if (save_gray_pgm(
                    run_directory,
                    "M",
                    frame,
                    gradient_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * E = thresholded edges
             * ---------------------------------------------------- */

            if (save_gray_pgm(
                    run_directory,
                    "E",
                    frame,
                    edge_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * R = legacy ROI
             * ---------------------------------------------------- */

            render_roi(
                &board_roi
            );


            if (save_gray_pgm(
                    run_directory,
                    "R",
                    frame,
                    line_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * L = Hough lines
             * ---------------------------------------------------- */

            render_hough_lines(
                line_count
            );


            if (save_gray_pgm(
                    run_directory,
                    "L",
                    frame,
                    line_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * C = Hough line families
             * ---------------------------------------------------- */

            render_hough_families(
                family_a_count,
                family_b_count
            );


            if (save_gray_pgm(
                    run_directory,
                    "C",
                    frame,
                    line_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * B = legacy Hough board outline
             * ---------------------------------------------------- */

            if (board_found) {

                render_board_outline(
                    &board
                );


                if (save_gray_pgm(
                        run_directory,
                        "B",
                        frame,
                        line_buffer,
                        IMG_WIDTH,
                        IMG_HEIGHT) == ESP_OK) {

                    saved_images++;
                }
            }


            /* ----------------------------------------------------
             * X = Harris corner candidates
             * ---------------------------------------------------- */

            render_chess_corners(
                chess_corner_count
            );


            if (save_gray_pgm(
                    run_directory,
                    "X",
                    frame,
                    line_buffer,
                    IMG_WIDTH,
                    IMG_HEIGHT) == ESP_OK) {

                saved_images++;
            }


            /* ----------------------------------------------------
             * Q = final detected 9x9 grid
             * ---------------------------------------------------- */

            if (grid_found) {

                render_chess_grid_debug(
                    &detected_grid,
                    line_buffer
                );


                if (save_gray_pgm(
                        run_directory,
                        "Q",
                        frame,
                        line_buffer,
                        IMG_WIDTH,
                        IMG_HEIGHT) == ESP_OK) {

                    saved_images++;
                }
            }


            /* ----------------------------------------------------
             * W = perspective-normalized chessboard
             *
             * IMPORTANT:
             * warp_chessboard() was already executed earlier.
             * We only SAVE its result here.
             * ---------------------------------------------------- */

            if (warp_ok) {

                if (save_gray_pgm(
                        run_directory,
                        "W",
                        frame,
                        board_warp_buffer,
                        BOARD_WARP_SIZE,
                        BOARD_WARP_SIZE) == ESP_OK) {

                    saved_images++;
                }
            }
        }
    }


    /* ============================================================
     * CLOSE CSV
     * ============================================================ */

    fclose(csv);


    /* ============================================================
     * SUMMARY FILE
     * ============================================================ */

    char summary_path[128];

    snprintf(
        summary_path,
        sizeof(summary_path),
        "%s/summary.txt",
        run_directory
    );


    FILE *summary =
        fopen(
            summary_path,
            "w"
        );


    if (summary != NULL) {


        /* --------------------------------------------------------
         * Average values
         * -------------------------------------------------------- */

        double average_capture_ms = 0.0;

        double average_extract_ms = 0.0;
        double average_gaussian_ms = 0.0;

        double average_sobel_ms = 0.0;
        double average_threshold_ms = 0.0;

        double average_dsp_ms = 0.0;
        double average_basic_pipeline_ms = 0.0;

        double average_chess_grid_ms = 0.0;
        double average_warp_ms = 0.0;

        double average_localization_ms = 0.0;

        double average_localization_pipeline_ms =
            0.0;

        double localization_fps = 0.0;


        if (successful_frames > 0) {


            average_capture_ms =
                (
                    (double)capture_total /
                    successful_frames
                ) /
                1000.0;


            average_extract_ms =
                (
                    (double)extract_total /
                    successful_frames
                ) /
                1000.0;


            average_gaussian_ms =
                (
                    (double)gaussian_total /
                    successful_frames
                ) /
                1000.0;


            average_sobel_ms =
                (
                    (double)sobel_total /
                    successful_frames
                ) /
                1000.0;


            average_threshold_ms =
                (
                    (double)threshold_total /
                    successful_frames
                ) /
                1000.0;


            average_dsp_ms =
                (
                    (double)dsp_total /
                    successful_frames
                ) /
                1000.0;


            average_basic_pipeline_ms =
                (
                    (double)pipeline_total /
                    successful_frames
                ) /
                1000.0;


            average_chess_grid_ms =
                (
                    (double)chess_grid_total /
                    successful_frames
                ) /
                1000.0;


            average_localization_ms =
                (
                    (double)localization_algorithm_total /
                    successful_frames
                ) /
                1000.0;


            average_localization_pipeline_ms =
                (
                    (double)localization_pipeline_total /
                    successful_frames
                ) /
                1000.0;


            if (average_localization_ms > 0.0) {

                localization_fps =
                    1000.0 /
                    average_localization_ms;
            }
        }


        /*
         * Perspective-warp average is calculated only over
         * successful warps.
         */
        if (warp_success_count > 0) {

            average_warp_ms =
                (
                    (double)warp_total /
                    warp_success_count
                ) /
                1000.0;
        }


        /* --------------------------------------------------------
         * Success percentages
         * -------------------------------------------------------- */

        double grid_success_percent =
            successful_frames > 0
                ? (
                    100.0 *
                    (double)grid_success_count /
                    successful_frames
                )
                : 0.0;


        double warp_success_percent =
            successful_frames > 0
                ? (
                    100.0 *
                    (double)warp_success_count /
                    successful_frames
                )
                : 0.0;


        /* --------------------------------------------------------
         * Write summary
         * -------------------------------------------------------- */

        fprintf(
            summary,

            "ESP32-CAM FINAL CHESSBOARD LOCALIZATION BENCHMARK\n"
            "================================================\n"
            "\n"

            "CONFIGURATION\n"
            "-------------\n"
            "Camera: OV2640\n"
            "Processor: ESP32 Xtensa LX6\n"
            "CPU frequency: 240 MHz\n"
            "Flash: 4 MB\n"
            "PSRAM: 2 MB\n"
            "Pixel format: YUV422\n"
            "Input resolution: %dx%d\n"
            "Normalized board: %dx%d\n"
            "Pixels per chess square: %dx%d\n"
            "\n"

            "TEST\n"
            "----\n"
            "Warm-up frames: %d\n"
            "Frames requested: %d\n"
            "Frames captured: %d\n"
            "Diagnostic images saved: %d\n"
            "\n"

            "FINAL LOCALIZATION SUCCESS\n"
            "--------------------------\n"
            "Grid detections: %d/%d (%.2f%%)\n"
            "Perspective warps: %d/%d (%.2f%%)\n"
            "\n"

            "FINAL LOCALIZATION TIMING\n"
            "-------------------------\n"
            "Average Y extraction: %.3f ms\n"
            "Average Gaussian 3x3: %.3f ms\n"
            "\n"

            "Average Harris + grid: %.3f ms\n"
            "Minimum Harris + grid: %.3f ms\n"
            "Maximum Harris + grid: %.3f ms\n"
            "\n"

            "Average perspective warp: %.3f ms\n"
            "Minimum perspective warp: %.3f ms\n"
            "Maximum perspective warp: %.3f ms\n"
            "\n"

            "Average localization algorithm: %.3f ms\n"
            "Minimum localization algorithm: %.3f ms\n"
            "Maximum localization algorithm: %.3f ms\n"
            "\n"

            "Average capture + localization: %.3f ms\n"
            "Effective localization rate: %.3f FPS\n"
            "\n"

            "CAMERA CAPTURE\n"
            "--------------\n"
            "Average framebuffer wait: %.3f ms\n"
            "Minimum framebuffer wait: %.3f ms\n"
            "Maximum framebuffer wait: %.3f ms\n"
            "\n"

            "LEGACY DSP COMPARISON\n"
            "---------------------\n"
            "Average Sobel: %.3f ms\n"
            "Average threshold: %.3f ms\n"
            "Average basic DSP total: %.3f ms\n"
            "Average capture + basic DSP: %.3f ms\n"
            "\n"

            "RESULT: %s\n",

            IMG_WIDTH,
            IMG_HEIGHT,

            BOARD_WARP_SIZE,
            BOARD_WARP_SIZE,

            BOARD_WARP_SIZE / 8,
            BOARD_WARP_SIZE / 8,

            WARMUP_FRAMES,

            TEST_FRAMES,

            successful_frames,

            saved_images,

            grid_success_count,

            successful_frames,

            grid_success_percent,

            warp_success_count,

            successful_frames,

            warp_success_percent,

            average_extract_ms,

            average_gaussian_ms,

            average_chess_grid_ms,

            chess_grid_min != INT64_MAX
                ? chess_grid_min / 1000.0
                : 0.0,

            chess_grid_max / 1000.0,

            average_warp_ms,

            warp_min != INT64_MAX
                ? warp_min / 1000.0
                : 0.0,

            warp_max / 1000.0,

            average_localization_ms,

            localization_min != INT64_MAX
                ? localization_min / 1000.0
                : 0.0,

            localization_max / 1000.0,

            average_localization_pipeline_ms,

            localization_fps,

            average_capture_ms,

            capture_min != INT64_MAX
                ? capture_min / 1000.0
                : 0.0,

            capture_max / 1000.0,

            average_sobel_ms,

            average_threshold_ms,

            average_dsp_ms,

            average_basic_pipeline_ms,

            (
                successful_frames ==
                    TEST_FRAMES &&

                grid_success_count ==
                    TEST_FRAMES &&

                warp_success_count ==
                    TEST_FRAMES
            )
                ? "PASS"
                : "PARTIAL"
        );


        fclose(summary);
    }


    /* ============================================================
     * FINAL CHECK
     * ============================================================ */

    if (successful_frames == 0) {

        ESP_LOGE(
            TAG,
            "No frames were captured"
        );

        return ESP_FAIL;
    }


    ESP_LOGI(
        TAG,
        "Benchmark complete: %d/%d frames",
        successful_frames,
        TEST_FRAMES
    );


    ESP_LOGI(
        TAG,
        "Grid success: %d/%d",
        grid_success_count,
        successful_frames
    );


    ESP_LOGI(
        TAG,
        "Warp success: %d/%d",
        warp_success_count,
        successful_frames
    );


    return ESP_OK;
}

/* MAIN */

void app_main(void)
{
    led_init();
    blink_once(150, 500);

    sdmmc_card_t *card = NULL;

    if (init_sd(&card) != ESP_OK) {
        blink_error_forever(3);
    }

    char run_directory[96];

    ESP_LOGI(TAG, "Creating run directory...");

    if (create_run_directory(run_directory, sizeof(run_directory)) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create run directory");
        blink_error_forever(3);
    }

    ESP_LOGI(TAG, "Run directory created: %s", run_directory);
    ESP_LOGI(TAG, "Writing status.txt...");

    if (write_status(run_directory, "SD_OK_CAMERA_NOT_STARTED") != ESP_OK) {
        blink_error_forever(4);
    }

    ESP_LOGI(TAG, "Finished status.txt step");
    ESP_LOGI(TAG, "Writing config.txt...");

    save_configuration(run_directory);

    ESP_LOGI(TAG, "Finished config.txt step");
    ESP_LOGI(TAG, "About to initialize camera");

    if (init_camera() != ESP_OK) {
        write_status(run_directory, "ERROR_CAMERA_INIT");
        blink_error_forever(5);
    }

    if (init_dsp_buffers() != ESP_OK) {
        write_status(run_directory, "ERROR_DSP_MEMORY");
        blink_error_forever(6);
    }

    if (init_hough() != ESP_OK) {
    write_status(run_directory, "ERROR_HOUGH_MEMORY");
    blink_error_forever(7); 
    }

    if (write_status(run_directory, "TEST_RUNNING") != ESP_OK) {
        blink_error_forever(4);
    }

    gpio_set_level(STATUS_LED_GPIO, 0);

    esp_err_t result = run_benchmark(run_directory);

    if (result != ESP_OK) {
        write_status(run_directory, "ERROR_DURING_BENCHMARK");
        blink_error_forever(4);
    }

    if (write_status(run_directory, "COMPLETE") != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write final COMPLETE status");
        blink_error_forever(4);
    }

    ESP_LOGI(TAG, "BENCHMARK completed successfully");
    blink_complete_forever();
}

// image_pthreads.c - Parallelized with pthreads
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NUM_THREADS 4

typedef struct {
    unsigned char *input;
    unsigned char *output;
    int width;
    int height;
    int channels;
    float *kernel;
    int kernel_size;
    int start_row;
    int end_row;
} thread_data_t;

// Filter kernels
float edge_kernel[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
float sharpen_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
float blur_kernel[9] = {1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9};
float gaussian_kernel[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
float emboss_kernel[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
float identity_kernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};

void *apply_convolution_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int kernel_half = data->kernel_size / 2;
    
    for (int y = data->start_row; y < data->end_row; y++) {
        for (int x = 0; x < data->width; x++) {
            for (int c = 0; c < data->channels; c++) {
                float sum = 0.0;
                
                // Apply kernel
                for (int ky = -kernel_half; ky <= kernel_half; ky++) {
                    for (int kx = -kernel_half; kx <= kernel_half; kx++) {
                        int img_y = y + ky;
                        int img_x = x + kx;
                        
                        // Handle borders by clamping
                        if (img_y < 0) img_y = 0;
                        if (img_y >= data->height) img_y = data->height - 1;
                        if (img_x < 0) img_x = 0;
                        if (img_x >= data->width) img_x = data->width - 1;
                        
                        int pixel_idx = (img_y * data->width + img_x) * data->channels + c;
                        int kernel_idx = (ky + kernel_half) * data->kernel_size + (kx + kernel_half);
                        
                        sum += data->input[pixel_idx] * data->kernel[kernel_idx];
                    }
                }
                
                // Clamp result to [0, 255]
                int output_idx = (y * data->width + x) * data->channels + c;
                data->output[output_idx] = (unsigned char)(fmax(0, fmin(255, sum)));
            }
        }
    }
    
    return NULL;
}

void apply_filter(unsigned char *input, unsigned char *output, int width, int height, 
                  int channels, float *kernel, int kernel_size) {
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    
    int rows_per_thread = height / NUM_THREADS;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].input = input;
        thread_data[i].output = output;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].channels = channels;
        thread_data[i].kernel = kernel;
        thread_data[i].kernel_size = kernel_size;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == NUM_THREADS - 1) ? height : (i + 1) * rows_per_thread;
        
        pthread_create(&threads[i], NULL, apply_convolution_thread, &thread_data[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <filter_type>\n", argv[0]);
        printf("Filter types: edge, sharpen, blur, gaussian, emboss, identity\n");
        return 1;
    }
    
    char *input_file = argv[1];
    char *filter_type = argv[2];
    
    int width, height, channels;
    unsigned char *img = stbi_load(input_file, &width, &height, &channels, 0);
    
    if (img == NULL) {
        printf("Error loading image %s\n", input_file);
        return 1;
    }
    
    printf("Loaded image: %dx%d with %d channels\n", width, height, channels);
    
    unsigned char *output = (unsigned char *)malloc(width * height * channels);
    
    float *kernel = NULL;
    int kernel_size = 3;
    
    if (strcmp(filter_type, "edge") == 0) {
        kernel = edge_kernel;
    } else if (strcmp(filter_type, "sharpen") == 0) {
        kernel = sharpen_kernel;
    } else if (strcmp(filter_type, "blur") == 0) {
        kernel = blur_kernel;
    } else if (strcmp(filter_type, "gaussian") == 0) {
        kernel = gaussian_kernel;
    } else if (strcmp(filter_type, "emboss") == 0) {
        kernel = emboss_kernel;
    } else if (strcmp(filter_type, "identity") == 0) {
        kernel = identity_kernel;
    } else {
        printf("Unknown filter type: %s\n", filter_type);
        stbi_image_free(img);
        free(output);
        return 1;
    }
    
    printf("Applying %s filter using pthreads with %d threads...\n", filter_type, NUM_THREADS);
    apply_filter(img, output, width, height, channels, kernel, kernel_size);
    
    stbi_write_png("output.png", width, height, channels, output, width * channels);
    printf("Output saved to output.png\n");
    
    stbi_image_free(img);
    free(output);
    
    return 0;
}
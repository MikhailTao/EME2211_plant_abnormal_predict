#include "C:\Users\Tao Ran\Desktop\EME2211\C,C++\EME2211\EasyBMP.h";
#include <math.h>;
#include <time.h>;
#include <cuda_runtime.h>;
#include <stdio.h>;
#include <stdlib.h>;

__global__ void polarize_G_8(const unsigned char* const R_in, const unsigned char* const G_in, const unsigned char* const B_in, unsigned char* const G_out, const int height, const int width) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; //assign thread
	((G_in[i] - R_in[i]) + (G_in[i] - B_in[i])) < 13 ? G_out[i] = 0 : G_out[i] = 255;
	//printf("G1[%d] = %d, G2[%d] = %d\n", i, color_in[i], i, color_out[i]);
}

__global__ void zero_8(const unsigned char* const color_in, unsigned char* const color_out, const int height, const int width) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; //assign thread
	color_out[i] = 0;
}

__global__ void max_8(const unsigned char* const color_in, unsigned char* const color_out, const int height, const int width) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; //assign thread
	color_out[i] = 255;
}

void readBMP(const char* FileName, uchar4** image_out, int* width, int* height) {
	BMP img;
	img.ReadFromFile(FileName);
	*width = img.TellWidth();
	*height = img.TellHeight();
	uchar4* const img_uchar4 = (uchar4*)malloc(*width * *height * sizeof(int));
	unsigned char r, g, b, a, i;
	// save each pixel to image_out as uchar4 in row-major format
	for (int row = 0; row < *height; row++)
		for (int col = 0; col < *width; col++) {
			img_uchar4[col + row * *width] = make_uchar4(img(col, row)->Red, img(col, row)->Green, img(col, row)->Blue, img(col, row)->Alpha);	//use row-major
		}
			

	*image_out = img_uchar4;
}

void writeBMP(const char* FileName, uchar4* image, int width, int height) {
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to the output image
	for (int row = 0; row < height; row++) {		//for each row
		for (int col = 0; col < width; col++) {	//for each col
			uchar4 rgba = image[col + row * width];
			output(col, row)->Red = rgba.x;
			output(col, row)->Green = rgba.y;
			output(col, row)->Blue = rgba.z;
			output(col, row)->Alpha = rgba.w;
		}
	}
	output.WriteToFile(FileName);

}

void checkForGPU() {
	// This code attempts to check if a GPU has been allocated
	// Colab notebooks technically have access to NVCC and will compile and
	// execute CPU/Host code, however, GPU/Device code will silently fail.
	// To prevent such situations, this code will warn the user.
	int count;
	cudaGetDeviceCount(&count);
	if (count <= 0 || count > 100) {
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		printf("->WARNING<-: NO GPU DETECTED ON THIS COLLABORATE INSTANCE.\n");
		printf("IF YOU ARE ATTEMPTING TO RUN GPU-BASED CUDA CODE, YOU SHOULD CHANGE THE RUNTIME TYPE!\n");
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	}
}

void polarize_G_32(const uchar4* const image_in, uchar4* const image_out, int height, int width) {
	//break the input image (uchar4 matrix) into 4 channels (four char matrices): Red, Green, Blue, and Alpha
	//in device
	unsigned char* d_R;
	unsigned char* d_G;
	unsigned char* d_B;
	unsigned char* d_A;
	//out device
	unsigned char* d_R2;
	unsigned char* d_G2;
	unsigned char* d_B2;
	unsigned char* d_A2;

	int size = width * height * sizeof(unsigned char);
	//allocate space on host;
	//in host
	unsigned char* R = (unsigned char*)malloc(size);
	unsigned char* G = (unsigned char*)malloc(size);
	unsigned char* B = (unsigned char*)malloc(size);
	unsigned char* A = (unsigned char*)malloc(size);
	//out host
	unsigned char* R2 = (unsigned char*)malloc(size);
	unsigned char* G2 = (unsigned char*)malloc(size);
	unsigned char* B2 = (unsigned char*)malloc(size);
	unsigned char* A2 = (unsigned char*)malloc(size);
	//in device
	cudaMalloc(&d_R, size);
	cudaMalloc(&d_G, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_A, size);
	//out device
	cudaMalloc(&d_R2, size);
	cudaMalloc(&d_G2, size);
	cudaMalloc(&d_B2, size);
	cudaMalloc(&d_A2, size);

	for (int i = 0; i < width * height; ++i) {	//break each pixel in input image
		uchar4 pxl = image_in[i];
		R[i] = pxl.x;
		G[i] = pxl.y;
		B[i] = pxl.z;
		A[i] = pxl.w;
	}

	cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	free(R); free(G); free(B); free(A);

	int nThreads = 1024;
	int nBlocks = height * width / nThreads;
	if ((height * width) % nThreads) nBlocks++;

	//perform 8-bit convolution for each 8-bit image channel 
	zero_8 << <nBlocks, nThreads >> > (d_R, d_R2, height, width);
	polarize_G_8 << <nBlocks, nThreads >> > (d_R, d_G, d_B, d_G2, height, width);
	zero_8 << <nBlocks, nThreads >> > (d_B, d_B2, height, width);
	max_8 << <nBlocks, nThreads >> > (d_A, d_A2, height, width);

	cudaFree(d_R); cudaFree(d_G); cudaFree(d_B); cudaFree(d_A);

	cudaMemcpy(R2, d_R2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(G2, d_G2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(B2, d_B2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(A2, d_A2, size, cudaMemcpyDeviceToHost);

	cudaFree(d_R2); cudaFree(d_G2); cudaFree(d_B2); cudaFree(d_A2);

	//merge the four channels into one output image of type uchar4
	for (size_t i = 0; i < height * width; ++i) {
		image_out[i] = make_uchar4(R2[i], G2[i], B2[i], A2[i]);
	}

	
	free(R2); free(G2); free(B2); free(A2);
}

void process() {

	const char* image_in_name = "test.bmp";
	const char* image_out_name = "test_G_P.bmp";

	//load input image
	int width, height;
	uchar4* image_in;
	readBMP(image_in_name, &image_in, &width, &height);	//image_in will have all pixel information, each pixel as uchar4
	printf("Input image loaded...\n");

	//apply convolution filter to input image
	uchar4* image_out = (uchar4*)malloc(width * height * sizeof(uchar4));	//reserve space in the memory for the output image
	printf("Processing...\n");
	int t = clock();
	polarize_G_32(image_in, image_out, height, width);	//filter applied to image_in, results saved in image_out
	t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
	printf("Polarizer applied. Time taken: %d.%d seconds\n", t / 1000, t % 1000);

	//save results to output image
	writeBMP(image_out_name, image_out, width, height);
	printf("Output image saved.\n");
}

void main() {
	//checkForGPU();
	clock_t start, t;
	start = clock();
	process();
	t = clock()-start;
	printf("Program finished! Time taken: %d.%d seconds\n", t / 1000, t % 1000);
}
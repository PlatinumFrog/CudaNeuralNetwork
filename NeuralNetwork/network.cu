#include "network.cuh"

__device__ float sigma(float x) {
	return x / (1 + ((x < 0) ? -x : x));
}

uint32_t nShift(uint32_t n) {
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}
/*
Parameters:
	Weights, Biases, Values, Weight Offsets, Bias Offsets, Value Offsets, Layer Number
*/
__global__ void calcNodes(float *w, float *b, float *v, uint32_t *wo, uint32_t *lo, uint32_t *ls, uint32_t *ln) {
	if(ln[0] > 0) {

		uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;

		// [0:         Weight Index Offset]
		// [1:  Current Layer Index Offset]
		// [2:          Current Layer Size]
		// [3: Previous Layer Index Offset]
		// [4:         Previous Layer Size]
		__shared__ size_t offsets[5];

		if(threadIdx.x == 0){
			offsets[0] = wo[ln[0]];
			offsets[1] = lo[ln[0]];
			offsets[2] = ls[ln[0]];
			offsets[3] = lo[ln[0] - 1];
			offsets[4] = ls[ln[0] - 1];
		}
		__syncthreads();

		if(id < offsets[2]) {
			float sum = 0;
			for(uint32_t i = 0; i < offsets[4]; i++) {
				sum += w[offsets[0] + ((id * offsets[4]) + i)] * v[offsets[3] + i];
			}
			v[offsets[1] + id] = sigma(sum + b[offsets[1] + id]);
		}
	}
}

color::color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	sr(r);
	sg(g);
	sb(b);
	sa(a);
}
color::color(uint32_t c) {
	this->col = c;
}

uint8_t color::gr() {
	return (uint8_t)(this->col & 255U);
}
uint8_t color::gg() {
	return (uint8_t)((this->col & (255U << 8U)) >> 8U);
}
uint8_t color::gb() {
	return (uint8_t)((this->col & (255U << 16U)) >> 16U);
}
uint8_t color::ga() {
	return (uint8_t)((this->col & (255U << 24U)) >> 24U);
}

void color::sr(uint8_t r) {
	this->col = (this->col & ~(255U) | (uint32_t)r);
}
void color::sg(uint8_t g) {
	this->col = ((this->col & ~(255U << 8U)) | (((uint32_t)g) << 8U));
}
void color::sb(uint8_t b) {
	this->col = ((this->col & ~(255U << 16U)) | (((uint32_t)b) << 16U));
}
void color::sa(uint8_t a) {
	this->col = ((this->col & ~(255U << 24U)) | (((uint32_t)a) << 24U));
}

bool color::operator==(color c) {
	return this->col == c.col;
}

Network::Network() : satisfaction(0) {
	


}

Network::Network(size_t nLayers, size_t sLayer) : satisfaction(0) {

	values.resize(nLayers * sLayer, 0);
	biases.resize(nLayers * sLayer, 0);
	weights.resize(nLayers * sLayer * sLayer, 0);
	currentLayer.resize(1);
	currentLayer[0] = 0;
	for(uint32_t i = 0; i < nLayers; i++) {
		weightOffsets.push_back((uint32_t)(i * sLayer * sLayer));
		layerOffsets.push_back((uint32_t)(i * sLayer));
		layerSizes.push_back((uint32_t)sLayer);
	}
}

Network::~Network() {
	
}

void Network::init() {

}

void Network::train() {}

void Network::run() {
	uint32_t blocks = 0;
	uint32_t threads = 0;
	for(uint32_t i = 1; i < layerSizes.size(); i++) {
		currentLayer[0] = i;
		uint32_t layerSize = layerSizes[i];
		if(layerSize > 0 && layerSize < 1024 ) {
			threads = nShift(layerSize);
			blocks = 1;
		}
		else if(layerSize > 1024) {
			threads = 1024;
			blocks = (uint32_t)ceil(layerSize / threads);
		}
		if(layerSize > 0) calcNodes<<<blocks,threads>>>(
			thrust::raw_pointer_cast(&weights[0]),
			thrust::raw_pointer_cast(&biases[0]),
			thrust::raw_pointer_cast(&values[0]),
			thrust::raw_pointer_cast(&weightOffsets[0]),
			thrust::raw_pointer_cast(&layerOffsets[0]),
			thrust::raw_pointer_cast(&layerSizes[0]),
			thrust::raw_pointer_cast(&currentLayer[0])
		);
		cudaDeviceSynchronize();
	}
}
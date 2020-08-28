#include "network.cuh"

__device__ float sigma(float x) {
	return x / (1 + ((x < 0) ? -x : x));
}

__global__ void init1(uint32_t *A, uint32_t size) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < size) A[id] = 1;
}

__global__ void initn(uint32_t *A, uint32_t size, uint32_t n) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < size) A[id] = n;
}

__global__ void initMult(uint32_t *A, uint32_t size, uint32_t mult) {
	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < size) A[id] = id * mult;
}

//Parameters: Weights, Biases, Values, Weight Offsets, Layer Offsets, Layer Sizes, Max ID
__global__ void calcAll(float *w, float *b, float *v, uint32_t *wo, uint32_t *lo, uint32_t *ls, uint32_t s) {

	uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id < s) {
		uint32_t ln = 0;
		while(id > lo[ln]) ln++;

		float sum = 0;
		for(uint32_t i = 0; i < ls[ln - 1]; i++) {
			sum += w[wo[ln] + ((id - lo[ln]) * ls[ln - 1]) + i] * v[lo[ln - 1] + i];
		}
		v[id] = sigma(sum + b[id]);
	}

}

Network::Network() : satisfaction(0), nLayers(0), isRunning(false) {}

Network::Network(uint32_t nl, uint32_t sl) : satisfaction(0), nLayers(nl), isRunning(false) {
	values.resize(nl * sl, 1);
	biases.resize(nl * sl, 1);
	weights.resize(nl * sl * sl, 1);
	layerOffsets.resize(nl, 0);
	layerSizes.resize(nl, sl);
	weightOffsets.resize(nl, 0);
	weightSizes.resize(nl, sl * sl);
	uint32_t blocks = 0, threads = 0;
	utility::findOccupancy(layerOffsets.size(), &blocks, &threads);
	initMult<<<blocks, threads>>>(thrust::raw_pointer_cast(&layerOffsets[0]), (uint32_t)layerOffsets.size(), sl);
	utility::findOccupancy(weightOffsets.size(), &blocks, &threads);
	initMult<<<blocks, threads>>>(thrust::raw_pointer_cast(&weightOffsets[0]), (uint32_t)layerOffsets.size(), sl * sl);
}

void Network::pushBack(uint32_t size) {

	if(nLayers > 0) {

		uint32_t pLayerSize = layerSizes.back();
		uint32_t pLayerOffset = layerOffsets.back();
		uint32_t pWeightSize = weightSizes.back();
		uint32_t pWeightOffset = weightOffsets.back();

		weightOffsets.push_back(pWeightOffset + pWeightSize);
		layerOffsets.push_back(pLayerOffset + pLayerSize);
		
		weightSizes.push_back(pLayerSize * size);
		layerSizes.push_back(size);

		values.resize(values.size() + size, 1);
		biases.resize(biases.size() + size, 1);
		weights.resize(weights.size() + (pLayerSize * size), 1);

	} else {

		layerSizes.push_back(size);
		layerOffsets.push_back(0);
		weightOffsets.push_back(0);
		weightSizes.push_back(0);
		values.resize(size,1);
		biases.resize(size,1);

	}
	nLayers++;
}

void Network::run() {
	while(isRunning){
		if(nLayers > 1) {
			uint32_t blocks = 0;
			uint32_t threads = 0;

			utility::findOccupancy(values.size(), &blocks, &threads);

			std::cout << "Exec: [" << values.size() << "]:[" << blocks << ", " << threads << "]:[";
			clock_t tb = clock();
			calcAll<<<blocks, threads>>>(
				thrust::raw_pointer_cast(&weights[0]),
				thrust::raw_pointer_cast(&biases[0]),
				thrust::raw_pointer_cast(&values[0]),
				thrust::raw_pointer_cast(&weightOffsets[0]),
				thrust::raw_pointer_cast(&layerOffsets[0]),
				thrust::raw_pointer_cast(&layerSizes[0]),
				(uint32_t)values.size()
			);
			std::cout << ((float)clock() - (float)tb) / CLOCKS_PER_SEC << "s]\n";
			
		}
	}
}

void Network::play() {
	if(!isRunning){
		isRunning = true;
		if(runThread.joinable()) {
			runThread.join();
			runThread.~thread();
		}
		runThread = std::thread(&Network::run, this);
	}
}

void Network::pause() {
	if(isRunning){
		isRunning = false;
		runThread.join();
	}
	cudaDeviceSynchronize();
}

Network::~Network() {
	cudaDeviceSynchronize();
	weightOffsets.clear();
	weightSizes.clear();
	layerOffsets.clear();
	layerSizes.clear();
	weights.clear();
	biases.clear();
	values.clear();
	weightOffsets.shrink_to_fit();
	weightSizes.shrink_to_fit();
	layerOffsets.shrink_to_fit();
	layerSizes.shrink_to_fit();
	weights.shrink_to_fit();
	biases.shrink_to_fit();
	values.shrink_to_fit();
	weightOffsets.~device_vector();
	weightSizes.~device_vector();
	layerOffsets.~device_vector();
	layerSizes.~device_vector();
	weights.~device_vector();
	biases.~device_vector();
	values.~device_vector();
}
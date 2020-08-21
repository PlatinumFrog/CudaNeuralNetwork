#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <curand_kernel.h>

struct color {

	uint32_t col;

	color(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
	color(uint32_t c);

	uint8_t gr();
	uint8_t gg();
	uint8_t gb();
	uint8_t ga();

	void sr(uint8_t r);
	void sg(uint8_t g);
	void sb(uint8_t b);
	void sa(uint8_t a);

	bool operator==(color c);

};

class Network {
private:

	thrust::device_vector<float> values;
	thrust::device_vector<float> biases;
	thrust::device_vector<float> weights;

	thrust::device_vector<uint32_t> weightOffsets;
	thrust::device_vector<uint32_t> layerOffsets;
	thrust::device_vector<uint32_t> layerSizes;
	thrust::device_vector<uint32_t> currentLayer;

	unsigned long long satisfaction;

public:

	Network();
	Network(size_t nLayers, size_t sLayer);

	~Network();

	void init();

	void train();

	void run();
};
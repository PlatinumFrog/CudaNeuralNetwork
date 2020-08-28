#include "utility.cuh"
#include "color.cuh"


class Network {
private:

	size_t satisfaction;

	size_t nLayers;
	
	bool isRunning;

	std::thread runThread;

	thrust::device_vector<float> values;
	thrust::device_vector<float> biases;
	thrust::device_vector<float> weights;

	thrust::device_vector<uint32_t> weightOffsets;
	thrust::device_vector<uint32_t> weightSizes;
	thrust::device_vector<uint32_t> layerOffsets;
	thrust::device_vector<uint32_t> layerSizes;

public:

	Network();

	Network(uint32_t nl, uint32_t sl);
	
	void pushBack(uint32_t size);

	void run();

	void play();

	void pause();
	
	~Network();
};
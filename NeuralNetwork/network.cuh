#include "utility.cuh"
#include "color.cuh"


class Network {
private:

	//std::string filename;

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
	//thrust::device_vector<uint32_t> currentLayer;

	/*void shiftLayers(uint32_t A, uint32_t B, uint32_t shiftAmount);

	void swapLayers(uint32_t A, uint32_t B);*/

	

public:

	Network();

	//Network(std::string filename);

	Network(uint32_t nl, uint32_t sl);
	
	void pushBack(uint32_t size);

	/*void load();

	void load(std::string filename);

	void save();

	void save(std::string filename);*/

	void run();

	/*void pullFront(uint32_t size);

	void add(uint32_t index, uint32_t size);

	void mod(uint32_t index, uint32_t size);

	void del(uint32_t index);

	void train();*/
	
	

	void play();

	void pause();
	
	~Network();
};
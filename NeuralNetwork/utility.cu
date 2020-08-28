#include "utility.cuh"

void utility::findOccupancy(size_t n, uint32_t *blocks, uint32_t *threads) {

	if(n > 0 && n < MAX_THREADS) {
		*threads = nShift((uint32_t)n);
		*blocks = 1;
	}
	else if(n > MAX_THREADS) {
		*threads = MAX_THREADS;
		*blocks = (uint32_t)ceil(n / *threads);
	}

}

uint32_t utility::nShift(uint32_t n) {
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}
#pragma once
#include "header.cuh"

#define MAX_THREADS 256

struct utility {
	static void findOccupancy(size_t n, uint32_t *blocks, uint32_t *threads);
	static uint32_t nShift(uint32_t n);
};
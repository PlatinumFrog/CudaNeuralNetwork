#pragma once
#include "header.cuh"
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
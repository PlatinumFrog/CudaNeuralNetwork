#include "color.cuh"

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
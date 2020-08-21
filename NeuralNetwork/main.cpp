#include "network.cuh"
#include "Header.h"

void run(Network *n, bool *stop) {
	clock_t tb;
	clock_t ta;
	while(*stop == false){
		tb = clock();
		n->run();
		ta = clock();
		std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n";
	}
};

int main() {
	std::cout << "-- Building! --\n";
	clock_t tb = clock();
	Network nn(1536, 1024);
	clock_t ta = clock();
	std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n";
	std::cout << "-- Running! --\n";
	bool stop = false;
	std::string s = "";
	std::thread t(run,&nn,&stop);
	std::cin >> s;
	stop = true;
	t.join();
	std::cout << "-- Deconstructing! --\n";
	tb = clock();
	nn.~Network();
	ta = clock();
	std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n-- Done! --\n";
	return 0;
}
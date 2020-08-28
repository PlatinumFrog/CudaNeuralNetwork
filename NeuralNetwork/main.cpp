#include "network.cuh"

int main() {

	std::cout << "-- Building! --\n";
	clock_t tb = clock();
	Network nn(1024,1024);
	/*for(uint16_t i = 0; i < 256; i++) {
		nn.pushBack(256);
	}*/
	clock_t ta = clock();
	std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n";


	std::cout << "-- Running! --\n";
	/*tb = clock();
	nn.run();
	ta = clock();
	std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n";*/

	std::string s;

	std::cin >> s;

	nn.play();

	std::cin >> s;

	nn.pause();

	std::cout << "-- Deconstructing! --\n";
	tb = clock();
	nn.~Network();
	ta = clock();
	std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n-- Done! --\n";

	return 0;
}
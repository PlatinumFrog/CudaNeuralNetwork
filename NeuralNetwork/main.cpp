#include "network.cuh"

int main() {

	std::cout << "-- Building! --\n";
	clock_t tb = clock();
	Network nn(1024,1024);
	clock_t ta = clock();
	std::cout << "Time:  " << ((float)ta - (float)tb) / CLOCKS_PER_SEC << "s\n";


	std::cout << "-- Running! --\n";

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
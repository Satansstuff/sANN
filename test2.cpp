#include "network.hpp"
#include <random>


int main()
{
	sa::net<double> neural;
	neural.construct(3,2,1);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-100, 100);
	unsigned numEpochs = 100000;
	double err = 0;
	unsigned epoch = 1;
	while(epoch < numEpochs)
	{
		auto l = dis(gen);
		std::vector<double> v = { 1.0, 1.0, 1.0 };
		std::vector<double> c = { 0.1 };
		neural.train(v, c);
		std::cout << "Epoch: " << epoch << std::endl;
		epoch++;
	}
	neural.feedForward(1.0,1.0,1.0);
	neural.saveToFile("derp.txt");
	std::cout << neural[0] << std::endl;;
	std::cin.ignore();
}

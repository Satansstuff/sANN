#include "network.hpp"
const double mutrate = 0.1;
int main()
{
	sa::batch<double> batch(100);
	batch.construct(3, 2, 1);
	std::vector<double> v, c;
	v.push_back(0.0);
	v.push_back(1.0);
	v.push_back(-1.0);
	c.push_back(1.0);
	while(true)
	{
		batch.trainBatch(v, c);
		batch = batch.mutate();
		if (batch.getMostFitNet().getAvgError() < 0.005)
			break;
		else
			std::cout << batch.getMostFitNet().getAvgError() << std::endl;
	}
	auto net = batch.getMostFitNet();
	net.feedForward(0.0, 1.0, -1.0);
	std::cout << net[0] << std::endl;
	std::cout << "AVG error: " << net.getAvgError() << std::endl;
	std::cin.ignore();
	return 0;
}

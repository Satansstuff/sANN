#include "network.hpp"
const double mutrate = 0.01;

int main()
{
	sa::batch<double> batch(20);
	batch.construct(3,2,1);
	std::vector<double> v,c;
	batch.trainBatch(v,c);
	return 0;
}
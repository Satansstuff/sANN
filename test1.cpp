#include "network.hpp"
const double mutrate = 0.01;

int main()
{
	sa::batch<double> batch(20);
	batch.construct(3,2,1);
	std::vector<double> v,c;
	v.push_back(0.0);
	v.push_back(1.0);
	v.push_back(-1.0);
	c.push_back(1.0);
	for(unsigned i = 0; i < 100; i++)
	{
		for(unsigned j = 0; j < 10; j++)
		{
			batch.trainBatch(v,c);
		}
		batch = batch.mutate();
	}
	return 0;
}
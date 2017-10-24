#include "network.hpp"



int main()
{
	sa::net<double> neural;
	neural.construct(3,8,2,1);
	std::vector<double> v = {1.0,1.0,-1.0};
	std::vector<double> c = {-1.0};
	neural.train(v,c);
	std::cout << neural[0] << std::endl;
}
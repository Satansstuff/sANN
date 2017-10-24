#include "network.hpp"



int main()
{
	sa::net<double> neural;
	neural.construct(3,8,2,1); 
	std::vector<double> v;
	v.push_back(1.0);
	v.push_back(2.0);
	v.push_back(3.0);
	std::vector<double> c;
	c.push_back(0.2);
	neural.train(v,c);
	std::cout << neural[0] << std::endl;
	std::cin.ignore();
}
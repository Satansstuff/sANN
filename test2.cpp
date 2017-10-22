#include "network.hpp"



int main()
{
	sa::net<double> neural;
	neural.construct(3,8,2,1); 
	neural.feedForward(5.0,2.0,1.0);
	std::cout << neural[0] << std::endl;
	std::cin.ignore();
}
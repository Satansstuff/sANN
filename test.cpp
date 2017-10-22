#include <iostream>
#include <windows.h>
#include <vector>
#include <iostream>
#include <chrono>
#include "network.hpp"
#define PRINTGEN 10000
#define max_gen 1000000000000
int main()
{
	std::srand(time_t(NULL));
	Perceptron<double, 3> p("new.txt");
	/*unsigned long long generation = 0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-50, 50);
	std::vector<double> values;
	values.resize(10);
	std::cout << "Training Perceptron: Press enter to move on" << std::endl;
	auto previousTime = std::chrono::steady_clock::now();
	while (!(GetKeyState(VK_ESCAPE) & 0x8000) || generation >= max_gen)
	{
		auto currentTime = std::chrono::steady_clock::now();
		auto elapsed = previousTime - currentTime;
		if (abs(elapsed.count() / 100000000) > 10)
		{
			previousTime = currentTime;
			std::cout << "Generation : " << generation << std::endl;
		}
		values.clear();
		double correct = 1.0;
		double val1 = dis(gen);
		double val2 = dis(gen);
		double val3 = dis(gen);
		values.push_back(val1);
		values.push_back(val2);
		values.push_back(val3);
		if (val3 <= 0.0)
		{
			correct = -1.0;
		}
		generation++;
		p.train(values, correct);
	}*/
	std::cout << "Done generating" << std::endl;
	while (true)
	{
		std::cout << "input three numbers: " << std::endl;
		double one, two, three;
		std::cin >> one >> two >> three;
		std::fflush(stdin);
		std::cout << p.feedForward(one, two, three) << std::endl;
		std::cin.ignore();
	}
	//p.saveToFile("new.txt");
	
	std::cin.ignore();
	std::cin.ignore();
	return EXIT_SUCCESS;
}
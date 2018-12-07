#include "network.hpp"
#include <unordered_map>
const double mutrate = 0.1;
std::vector<std::vector<double>> emails;
std::vector<std::vector<double>> classified;
int main()
{
	std::ifstream f("spambase.data");
	if (f.is_open())
	{
		std::string str;
		while (std::getline(f, str))
		{
			std::stringstream ss(str);
			std::vector<double> m_f;
			std::vector<double> email;
			double i;
			int counter = 0;
			while (ss >> i)
			{
				email.push_back(i);
				if (ss.peek() == ',')
					ss.ignore();
				counter++;
			}
			m_f.push_back(email.back());
			classified.push_back(m_f);
			email.pop_back();
			emails.push_back(email);
		}
	}
	else
	{
		std::cerr << "failed" << std::endl;
		std::cin.ignore();
		std::exit(-1);
	}
	sa::batch<double> batch(10);
	batch.construct(emails[0].size(), 512, 256, 128, 1);
	long generation = 0;
	double prevError = 1;
	while(true)
	{
		for (unsigned i = 0; i < emails.size(); i++)
		{
			batch.trainBatch(emails[i], classified[i]);
		}
		auto net = batch.getMostFitNet();
		auto error = net.getAvgError();
		if (error < 0.00000000000001)
		{
			std::cout << error << std::endl;
			break;
		}
		std::cout << "Generation: " << generation << " AvgError: " << error << std::endl;
		if (error < prevError)
		{
			batch = sa::batch<double>(10, net);
			prevError = error;
		}
		generation++;
	}
	auto net = batch.getMostFitNet();
	net.saveToFile("trainednet.net");
	unsigned correct = 0, incorrect = 0;
	for (unsigned i = 0; i < emails.size(); i++)
	{
		net.feedForward(emails[i]);
		bool spam = false;
		if (net[0] > 0.5)
		{
			spam = true;
		}
		else
		{
			spam = false;
		}
		if (classified[i][0] == (double)spam)
		{
			correct++;
		}
		else
		{
			incorrect++;
		}
	}
	std::cout << "Correctly predicted: " << correct << std::endl;
	std::cout << "Incorrectly predicted: " << incorrect << std::endl;
	std::cin.ignore();
	return 0;
}
#pragma once
#include <random>
#include <string>
#include <fstream>
#include <exception>
#include <array>
#include <iostream>
#include <cmath>
#include <functional>
#include <ctime>
#include <memory>
/*
Varadic templates
Gogo psykos
Git aids
Otränad aids, typ som jag.
*/
namespace sa
{
	struct neuron;
	template <typename T>
	class net;
	typedef std::shared_ptr<neuron> neuronPointer;
	typedef std::vector<neuronPointer> layer;
	template <typename T>
	static T fsigm(T f)
	{
		return 1.0 / (1.0 + exp(-f));
	}
	struct neuron
	{
		std::vector<double> m_weights;
		std::vector<double> m_deltaWeights;
		double bias = 1;
		double output = 0;
		double gain;
		double wgain;
		double lc;
		neuron(size_t num_weights, double learning = 0.01)
		{
			m_weights.reserve(num_weights);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<double> dis(-2.4 / num_weights, 2.4 / num_weights);
			for (size_t i = 0; i < num_weights; i++)
			{
				m_weights.push_back(dis(gen));
			}
			lc = learning;
		}
		void feedForward(layer &prevLayer)
		{
			double sum = bias;
			for (auto neuron : prevLayer)
			{
				for (auto &weight : neuron->m_weights)
				{
					sum += neuron->output * weight;
				}
			}
			output = fsigm<double>(sum);
		}
		void evolve(double mutRate)
		{
			for (auto &w : m_weights)
			{
				w *= mutRate * lc;
			}
		}
	};
	template <typename T>
	class net
	{
	private:
		bool initialized;
		std::vector<layer> m_layers;
	public:
		void saveToFile(const std::string &f)
		{

		}
		void loadFromFile(const std::string &f)
		{

		}
		void evolve()
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dis(-1, 1);
			for (size_t i = 1; i < m_layers.size(); i++)
			{
				for (size_t j = 0; j < m_layers[i].size(); j++)
				{
					m_layers[i][j]->evolve(dis(gen));
				}
			}
		}
		double operator[](size_t i)
		{
			if (i > m_layers.back().size() || i < 0)
			{
				std::runtime_error("No outputnode found");
			}
			return m_layers.back()[i]->output;
		}
		template <class... Params>
		void construct(Params... params)
		{
			std::array<size_t, sizeof...(params)> layers = { (size_t)params... };
			const size_t n = sizeof...(Params);
			m_layers.resize(n);
			for (size_t i = 0; i < n; i++)
			{
				m_layers[i].resize(layers[i]);
			}
			for (size_t i = 0; i < n; i++)
			{
				for (size_t j = 0; j < layers[i]; j++)
				{
					if (i != 0)
					{
						m_layers[i][j] = std::make_shared<neuron>(neuron(m_layers[i - 1].size()));
					}
					else
					{
						m_layers[i][j] = std::make_shared<neuron>(neuron(1));
					}
				}
			}
			initialized = true;
		}
		net()
		{
			initialized = false;
		}
		void train(std::vector<T> &values, std::vector<T> &expected)
		{
			if(values.size() != m_layers[0].size())
			{
				std::runtime_error("More/less values than input-nodes");
			}
			if(expected.size() != m_layers.back().size())
			{
				std::runtime_error("More/less values than exit-nodes");
			}
			//FeedForward
			for (size_t i = 0; i < m_layers[0].size(); i++)
			{
				m_layers[0][i]->output = values[i];
			}
			for (size_t i = 1; i < m_layers.size(); i++)
			{
				for (size_t j = 0; j < m_layers[i].size(); j++)
				{
					m_layers[i][j]->feedForward(m_layers[i - 1]);
				}
			}
			std::vector<T> outValues;
			for(size_t i = 0; i < m_layers.back().size(); i++)
			{
				outValues[i] = m_layers.back()[i]->output;
			}
			//calculate-errors?
			//BackPropogate
			//Update Weights
			//????

		}
		template <class... Params>
		void feedForward(Params... params)
		{
			std::array<T, sizeof...(params)> values = { (T)params... };
			static const std::size_t value = sizeof...(params);
			if (value > m_layers[0].size())
			{
				std::runtime_error("More values than input-neurons");
			}
			else if (value < m_layers[0].size())
			{
				std::runtime_error("Fewer values than input-neurons");
			}
			for (size_t i = 0; i < m_layers[0].size(); i++)
			{
				m_layers[0][i]->output = values[i];
			}
			for (size_t i = 1; i < m_layers.size(); i++)
			{
				for (size_t j = 0; j < m_layers[i].size(); j++)
				{
					m_layers[i][j]->feedForward(m_layers[i - 1]);
				}
			}
		}

	};

	/*
	Perceptron är basically ett network med bara en neuron
	Supervised learning
	Genetic learning?
	*/
	template <typename T, int numVals>
	class Perceptron
	{
	private:
		unsigned counter = 0;
		T feedValue;
		T InternalfeedForward(std::vector<T> values)
		{
			T sum = 1;
			for (unsigned i = 0; i < m_weights.size(); i++)
			{
				sum += m_weights[i] * values[i];
			}
			return fsigm(sum);
		}
	protected:
		unsigned numberOfVals;
		double lrate;
		std::vector<T> m_weights;
	public:
		Perceptron(const std::string &file, const double _lrate = 0.02)
		{
			numberOfVals = numVals;
			std::ifstream f(file);
			if (!f)
			{
				std::runtime_error("Failed to load-file \n");
			}
			std::string line;
			while (std::getline(f, line))
			{
				m_weights.push_back(atof(line.c_str()));
			}
			f.close();
			lrate = _lrate;
		}
		void loadFromFile(const std::string &file)
		{
			std::ifstream f(file);
			if (!f)
			{
				std::runtime_error("Failed to load-file \n");
			}
			std::string line;
			while (std::getline(f, line))
			{
				m_weights.push_back(atof(line.c_str()));
			}
			f.close();
		}
		void saveToFile(const std::string &file)
		{
			std::ofstream f(file, std::ofstream::out);
			if (!f)
			{
				std::runtime_error("Failed to Open-file");
			}
			for (size_t i = 0; i < m_weights.size(); i++)
			{
				f << std::to_string(m_weights[i]) << std::endl;
			}
			f.close();
		}
		Perceptron(const double _lrate = 0.02)
		{
			numberOfVals = numVals;
			srand((unsigned)time(NULL));
			lrate = _lrate;
			m_weights.reserve(numVals);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dis(-2.4 / numVals, 2.4 / numVals);
			for (size_t i = 0; i < numVals; i++)
			{
				m_weights.push_back(dis(gen));
			}
		}
		Perceptron(std::vector<T> weights, const double _lrate = 0.1)
		{
			numberOfVals = numVals;
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dis(-1, 1);
			for (size_t i = 0; i < m_weights.size(); i++)
			{
				m_weights[i] += weights[i] * lrate * dis(gen); //Mutera
			}
			lrate = _lrate;
		}
		void train(std::vector<T> values, T corr)
		{
			T correct = corr;
			T guess = InternalfeedForward(values);
			T error = correct - guess;
			for (unsigned j = 0; j < m_weights.size(); j++)
			{
				m_weights[j] += lrate * error * values[j];
			}
		}
		T feedForward(T first)
		{
			if (counter < m_weights.size())
			{
				return feedForward(first * m_weights[counter++]);
			}
			else
			{
				return feedValue;
			}
		}
		template <typename... Args>
		T feedForward(T first, Args... A)
		{
			static const std::size_t value = sizeof...(Args);
			static_assert(value + 1 <= numVals, "Too many values");
			feedValue = first * m_weights[counter] + 1;
			counter = 1;
			feedValue = feedForward(A...);
			counter = 0;
			T sig = net<T>::fsigm(feedValue);
			feedValue = 0;
			return sig;
		}
	};
}
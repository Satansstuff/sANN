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
Otr�nad aids, typ som jag.
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
		T ret = 1.0 / (1.0 + std::exp(-f));
		return ret;
	}
	inline static int32_t clampOutputValue(double x)
	{
		if (x < 0.1) return 0;
		else if (x > 0.9) return 1;
		else return -1;
	}
	template <typename T>
	static T dfsigm(T f)
	{
		return f * (1 - f);
	}
	struct neuron
	{
		std::vector<double> m_weights;
		std::vector<double> m_deltaWeights;
		double bias = 1;
		double output = 0;
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
		T operator[](size_t i)
		{
			if (i > m_layers.back().size() || i < 0)
			{
				std::runtime_error("No outputnode found");
			}
			T output = m_layers.back()[i]->output;
			return output;
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
			layer results = this->feedForward(values);
			std::vector<T> resultValues;
			std::vector<T> outErrors;
			outErrors.resize(expected.size());
			for (auto &result : results)
			{
				resultValues.push_back(result->output);
			}
			if (resultValues.size() != expected.size())
			{
				std::runtime_error("Length-mismatch");
			}
			//Output-layer
			for (size_t i = 0; i < expected.size(); i++)
			{
				double error = expected[i] - resultValues[i];
				for (auto &neuron : results)
				{
					for (unsigned j = 0; j < neuron->m_weights.size(); j++)
					{
						neuron->m_weights[j] += 0.1 * error * dfsigm(neuron->output);
					}
				}
			}
		}
		layer feedForward(std::vector<T> &values) &
		{ 
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
			return m_layers.back();
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
}

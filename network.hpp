/*
Created by Satan
Give respect where respect is due, without me; There would be no you.
Feel free to use this wherever you seem fit, but do give me a mention somewhere.
Also if this becomes skynet, you owe me a beer.
*/


/*
	Multi-threading would be nice...
*/
#pragma once
#include <random>
#include <exception>
#include <string>
#include <fstream>
#include <streambuf>
#include <array>
#include <iostream>
#include <cmath>
#include <functional>
#include <map>
#include <sstream>
#include <ctime>
#include <memory>
#include <vector>
#include <thread>
#include <future>


std::vector<std::string> split(std::string target, char delim)
{
	std::vector<std::string> lk;
	std::istringstream ss(target);
	std::string line;
	while (std::getline(ss, line, delim))
	{
		lk.push_back(line);
	}
	return lk;
}
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
	struct neuron
	{
		std::vector<double> m_weights;
		double bias = 1;
		double output = 0;
		double delta = 0;
		double lc = 0.01;
		neuron(size_t num_weights, double learning = 0.01)
		{
			m_weights.reserve(num_weights);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<double> dis(-3, 3);
			for (size_t i = 0; i < num_weights; i++)
			{
				m_weights.push_back(dis(gen));
			}
			bias = dis(gen);
			lc = learning;
		}
		void evolve(double rate)
		{
			for (auto &w : m_weights)
			{
				w += rate * lc;
			}
		}
		neuron()
		{
			//SHOULD I EVEN EXIST?
		}
		void updateFreeParams(layer &prevlayer)
		{
			bias = bias + lc * 1 * delta;
			for (size_t i = 0; i < prevlayer.size(); i++)
			{
				m_weights[i] += lc * prevlayer[i]->output * delta;
			}
		}
		void feedForward(layer &prevLayer)
		{
			double sum = bias * 1;
			for (size_t i = 0; i < prevLayer.size(); i++)
			{
				sum += prevLayer[i]->output * m_weights[i];
			}
			output = fsigm<double>(sum);
		}
		friend std::ostream& operator<<(std::ostream& os, const neuron& n)
		{
			os << n.bias << " " << n.delta << " " << n.lc << " " << n.output << " " << n.m_weights.size() << " ";
			for (size_t i = 0; i < n.m_weights.size(); i++)
			{
				os << n.m_weights[i];
			}
			return os;
		}
		friend std::istream& operator>> (std::istream& is, neuron& n)
		{
			std::string crap;
			size_t numWeights = 1;
			is >> n.bias >> n.delta >> n.lc >> n.output >> numWeights;
			n.m_weights.resize(numWeights);
			for (size_t i = 0; i < n.m_weights.size(); i++)
			{
				is >> n.m_weights[i];
			}
			return is;
		}
	};
	template <typename T>
	class net
	{
	private:
		bool initialized;
		std::vector<layer> m_layers;
	public:
		net(const net<T> &other)
		{
			std::vector<layer> layers = other.m_layers;
			m_layers.resize(layers.size());
			for (size_t i = 0; i < layers.size(); i++)
			{
				layer newLayer;
				for (auto &n : layers[i])
				{
					neuron newNeron(*n);
					newLayer.push_back(std::make_shared<neuron>(newNeron));
				}
				m_layers[i] = newLayer;
			}
		}
		T getAvgError()
		{
			T errorSum = 0;
			layer l = m_layers.back();
			for (auto &OutNeuron : l)
			{
				auto error = abs(OutNeuron->delta / (OutNeuron->output * (1 - OutNeuron->output)));
				errorSum += error;
			}
			return errorSum / m_layers.back().size();
		}
		net operator=(const net &other)
		{
			std::vector<layer> layers = other.m_layers;
			m_layers.resize(layers.size());
			for (size_t i = 0; i < layers.size(); i++)
			{
				layer newLayer;
				for (auto &n : layers[i])
				{
					neuron newNeron(*n);
					newLayer.push_back(std::make_shared<neuron>(newNeron));
				}
				m_layers[i] = newLayer;
			}
			return *this;
		}
		void mutate(const double mutrate)
		{
			std::default_random_engine generator;
			std::uniform_int_distribution<int> distribution(-1, 1);
			for (auto &layer : m_layers)
			{
				for (auto &neuron : layer)
				{
					neuron->evolve(mutrate * distribution(generator));
				}
			}
		}
		void saveToFile(const std::string &f)
		{
			std::ofstream file(f);
			//Number of layers in total
			file << m_layers.size() << std::endl;
			//Write input-neurons
			file << m_layers[0].size() << ":";
			for (size_t i = 0; i < m_layers[0].size(); i++)
			{
				file << *m_layers[0][i].get();
			}
			file << std::endl;
			size_t numHidden = m_layers.size() - 2;
			//Write Hidden-layers
			for (size_t i = 1; i <= numHidden; i++)
			{
				file << m_layers[i].size() << ":";
				for (size_t j = 0; j < m_layers[i].size(); j++)
				{
					file << *m_layers[i][j].get();
				}
				file << std::endl;
			}
			//Write Output-Layer
			file << m_layers.back().size() << ":";
			for (size_t i = 0; i < m_layers.back().size(); i++)
			{
				file << *m_layers.back()[i].get();
			}

		}
		void loadFromFile(const std::string &f)
		{
			std::ifstream t(f);
			std::string str;

			t.seekg(0, std::ios::end);
			str.reserve((size_t)t.tellg());
			t.seekg(0, std::ios::beg);

			str.assign((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());

			auto layers = split(str, '\n');
			m_layers.resize(stoi(layers[0]));
			layers.erase(layers.begin());
			for (size_t i = 0; i < m_layers.size(); i++)
			{
				std::istringstream is(layers[i]);
				neuron buffer;
				while (is >> buffer)
				{
					m_layers[i].push_back(std::make_shared<neuron>(buffer));
				}
			}
		}
		layer back() &
		{
			return m_layers.back();
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
		net(const std::string &str)
		{
			initialized = false;
		}
		void train(std::vector<T> &values, std::vector<T> &expected)
		{
			layer results = this->feedForward(values);
			std::vector<T> resultValues;
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
					for (size_t j = 0; j < neuron->m_weights.size(); j++)
					{
						neuron->delta = neuron->output * (1 - neuron->output) * error;
					}
				}
			}
			//Hidden-Layers
			size_t numHiddenLayers = m_layers.size() - 2;
			for (size_t i = numHiddenLayers; i > 0; i--)
			{
				double error = 0.0;
				layer currLayer = m_layers[i];
				layer topLayer = m_layers[i + 1];
				layer prevLayer = m_layers[i - 1];
				for (auto &n : currLayer)
				{
					for (size_t j = 0; j < topLayer.size(); j++)
					{
						error += topLayer[j]->delta * n->m_weights[prevLayer.size() - 1];
					}
					n->delta = n->output * (1 - n->output) * error;
				}
			}
			//Update Weights
			for (size_t i = m_layers.size() - 1; i > 0; i--)
			{
				for (auto &n : m_layers[i])
				{
					n->updateFreeParams(m_layers[i - 1]);
				}
			}
			return;
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
	template <typename T>
	class batch
	{
	private:
		bool initialized = false;
		std::vector<net<T>> m_nets;
	public:
		batch(size_t batchSize)
		{
			m_nets.resize(batchSize);
		}
		batch(size_t batchSize, sa::net<T> &templat)
		{
			m_nets.resize(batchSize);
			for (unsigned i = 0; i < batchSize; i++)
			{
				m_nets[i] = sa::net<T>(templat);
				m_nets[i].mutate(0.02);
			}
		}
		size_t size()
		{
			return m_nets.size();
		}
		batch mutate(size_t batchSize = 0, double mutateRate = 0.02)
		{
			if (batchSize == 0)
			{
				batchSize = this->size();
			}
			if (this->size() < 1)
			{
				std::runtime_error("Batch empty");
			}
			batch newB(*this);
			newB.m_nets.resize(batchSize);
			net<T> bestNet = getMostFitNet();
			for (size_t i = 0; i < batchSize; i++)
			{
				newB.m_nets[i] = net<T>(bestNet);
				newB.m_nets[i].mutate(mutateRate);
			}
			if (newB.getMostFitNet().getAvgError() < this->getMostFitNet().getAvgError())
			{
				return newB;
			}
			else
			{
				return *this;
			}
		}
		template <class... Params>
		void construct(Params... p)
		{
			for (auto &net : m_nets)
			{
				net.construct(p...);
			}
			initialized = true;
		}
		void trainBatch(std::vector<T> &values, std::vector<T> &expected)
		{
			if (!initialized)
				std::runtime_error("Batch not initialized!");
			std::vector<std::future<void>> m_futures;
			m_futures.resize(m_nets.size() - 1);
			for (unsigned i = 0; i < m_nets.size() - 1; i++)
			{
				m_futures[i] = std::async(std::launch::async, [&] { return m_nets[i].train(values, expected); });
			}
			for (unsigned i = 0; i < m_futures.size(); i++)
			{
				m_futures[i].wait();
			}
		}
		net<T> getMostFitNet()
		{
			std::vector<T> errorSums;
			errorSums.resize(m_nets.size());
			for (size_t i = 0; i < m_nets.size(); i++)
			{
				layer l = m_nets[i].back();
				for (auto &OutNeuron : l)
				{
					auto error = abs(OutNeuron->delta / (OutNeuron->output * (1 - OutNeuron->output)));
					errorSums[i] += error;
				}
			}
			size_t index = 0;
			T smallest = INFINITY;
			for (size_t i = 0; i < errorSums.size(); i++)
			{
				if (errorSums[i] < smallest)
				{
					smallest = errorSums[i];
					index = i;
				}
			}
			return m_nets[index];
		}
	};
}

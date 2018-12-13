#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>
#include <future>
#include <array>
namespace sANN
{
	template<typename T>
	struct dataset
	{
	private:
		size_t count_line(std::istream &is)
		{
			if (is.bad()) return 0;
			std::istream::iostate state_backup = is.rdstate();
			is.clear();
			std::istream::streampos pos_backup = is.tellg();
			is.seekg(0);
			size_t line_cnt;
			size_t lf_cnt = std::count(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>(), '\n');
			line_cnt = lf_cnt;
			is.unget();
			if (is.get() != '\n') { ++line_cnt; }
			is.clear();
			is.seekg(pos_backup);
			is.setstate(state_backup);

			return line_cnt;
		}
	public:
		std::vector<std::vector<T>> data;
		std::vector<std::vector<T>> training;
		unsigned falsePositive = 0, falseNegative = 0, truePositive = 0, trueNegative = 0;
		void clearTrainingdata()
		{
			training.clear();
		}
		int64_t size()
		{
			return data[0].size() - 1;
		}
		dataset()
		{
			
		}
		dataset(const std::string path, size_t size = 0, char delim = ' ', bool isTraining = false, unsigned numTrainingVar = 1)
		{
			std::ifstream dataset(path);
			std::string str;
			if (dataset.is_open())
			{
				if (size > 0)
				{
					data.resize(size);
				}
				if (isTraining)
				{
					training.resize(count_line(dataset));
				}
				uint64_t counter = 0;
				while (std::getline(dataset, str))
				{
					std::stringstream ss(str);
					T i;
					data.push_back(std::vector<T>());
					while (ss >> i)
					{
						data.back().push_back(i);
						if (ss.peek() == delim)
						{
							ss.ignore();
						}
					}
					if (isTraining)
					{
						for (unsigned k = 0; k < numTrainingVar; k++)
						{
							T c = data.back().back();
							data.back().pop_back();
							training[counter].push_back(c);
						}
					}
					counter++;
				}

			}
			else
			{
				std::runtime_error("Failed to open dataset");
			}
		};
	};
	struct neuron;
	class network;
	enum activationFunctions
	{
		sigm,
		identity,
		gaussian,
		num_functions
	};
	namespace internals
	{
		bool initialized = false;
		std::uniform_real_distribution<double> unif(-3, 3);
		std::default_random_engine re;
		std::unordered_map<unsigned int, std::function<double(double)>> m_activationfunctions;
		std::unordered_map<unsigned int, std::function<double(double)>> m_derivedActivationfunctions;
		static double sigm(double f)
		{
			auto ret = 1.0 / (1.0 + std::exp(-f));
			return ret;
		}
		static double dSigm(double f)
		{
			return sigm(f) * (1 - sigm(f));
		}
		static double identity(double f)
		{
			return f;
		}
		static double dIdentity(double f)
		{
			return 1;
		}
		static double gaussian(double f)
		{
			return std::exp(-f * -f);
		}
		static double dGaussian(double f)
		{
			return -2 * f * std::exp(-f * -f);
		}
		void init()
		{
			srand((unsigned)std::time(NULL));
			m_activationfunctions[activationFunctions::sigm] = sigm;
			m_activationfunctions[activationFunctions::identity] = identity;
			m_activationfunctions[activationFunctions::gaussian] = gaussian;

			m_derivedActivationfunctions[activationFunctions::sigm] = dSigm;
			m_derivedActivationfunctions[activationFunctions::identity] = dSigm;
			m_derivedActivationfunctions[activationFunctions::gaussian] = dGaussian;
			if (m_activationfunctions.size() == activationFunctions::num_functions && m_derivedActivationfunctions.size() == activationFunctions::num_functions)
			{
				initialized = true;
			}
		}
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
	};
	typedef std::shared_ptr<neuron> neuronPointer;
	typedef std::vector<neuronPointer> layer;
	struct neuron
	{
		std::vector<double> m_weights;
		std::vector<neuron> m_neighbors;
		double output;
		double bias;
		double delta = 0;
		double lc = 0.01;
		neuron()
		{
			//Needed for saving/loading :>
		}
		neuron(size_t inputs)
		{
			bias = internals::unif(internals::re);
			m_weights.resize(inputs);
			for (size_t i = 0; i < inputs; i++)
			{
				m_weights[i] = internals::unif(internals::re);
			}
		}
		void updateWeights(layer &prevLayer, std::function<double(double)> &func)
		{
			bias = bias + lc * delta;
			for (size_t i = 0; i < prevLayer.size(); i++)
			{
				m_weights[i] += func(output) * delta * lc;
			}
			output = 0;
			delta = 0;
		}
		void feedForward(layer &prevLayer, std::function<double(double)> &func)
		{
			double sum = bias * 1;
			for (size_t i = 0; i < prevLayer.size(); i++)
			{
				sum += prevLayer[i]->output * m_weights[i];
			}
			output = func(sum);
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
	class network
	{
	private:
		std::function<double(double)> m_activationfunction;
		std::function<double(double)> m_derivationFunction;
		bool initialized = false;
		std::vector<layer> m_layers;
		template<typename T>
		layer feedForward(std::vector<T> data)
		{
			if (data.size() != m_layers[0].size())
			{
				std::runtime_error("Too few arguments");

			}
			auto feed = [this](size_t &i)
			{
				for (size_t j = 0; j < m_layers[i].size(); j++)
				{
					m_layers[i][j]->feedForward(m_layers[i - 1], m_activationfunction);
				}
			};
			std::vector<std::future<void>> m_futures;
			for (size_t i = 0; i < m_layers[0].size(); i++)
			{
				m_layers[0][i]->output = data[i];
			}
			for (size_t i = 1; i < m_layers.size(); i++)
			{
				m_futures.resize(m_layers[i].size());
				for (unsigned e = 0; e < m_futures.size(); e++)
				{
					m_futures[e] = std::async(std::launch::async, [&] { feed(i); });
				}
				for (unsigned e = 0; e < m_futures.size(); e++)
				{
					m_futures[e].wait();
				}
				
			}
			return m_layers.back();
		}
	public:

		network(unsigned func = activationFunctions::sigm)
		{
			if (!internals::initialized)
			{
				internals::init();
				m_activationfunction = internals::m_activationfunctions[func];
				m_derivationFunction = internals::m_derivedActivationfunctions[func];
			}
			initialized = false;
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
		template <class... Params>
		network(Params... params)
		{
			if (!internals::initialized)
			{
				internals::init();
				m_activationfunction = internals::m_activationfunctions[activationFunctions::sigm];
				m_derivationFunction = internals::m_derivedActivationfunctions[activationFunctions::sigm];
			}
			std::array<size_t, sizeof...(params)> layers = { (size_t)params... };
			const size_t n = sizeof...(Params);
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
		double getAvgError()
		{
			double errorSum = 0;
			layer l = m_layers.back();
			for (auto &OutNeuron : l)
			{
				auto error = abs(OutNeuron->delta / (OutNeuron->output * (1 - OutNeuron->output)));
				errorSum += error;
			}
			return errorSum / m_layers.back().size();
		}
		template<typename T>
		double calculateTotalError(std::vector<T> &set)
		{
			double error = 0.0;
			for (size_t i = 0; i < m_layers.back().size(); i++)
			{
				error += 1 / 2 * std::pow((set[i] - m_layers.back()[i]), 2);
			}
			return error;
		}
		template<typename T>
		void train(sANN::dataset<T> &dataset)
		{
			if (!initialized)
			{
				std::runtime_error("Network not initialized");
				return;
			}
			if (dataset.training.size() == 0)
			{
				std::runtime_error("No training data given.");
				return;
			}
			std::vector<T> resultValues;
			std::vector<layer> results;
			for (unsigned i = 0; i < dataset.data[i].size(); i++)
			{
				results.push_back(this->feedForward(dataset.data[i]));
				for (auto &result : results[i])
				{
					resultValues.push_back(result->output);
				}
			}
			if (dataset.data.size() != dataset.training.size())
			{
				std::runtime_error("Length-mismatch");
			}
			for (size_t h = 0; h < dataset.training.size() - 1; h++)
			{
				//Output-layer
				for (size_t i = 0; i < dataset.training[h].size(); i++)
				{
					for (unsigned j = 0; j < results.size(); j++)
					{
						for (auto &neuron : results[j])
						{
							double error = (dataset.training[h][i] - m_layers.back()[i]->output) * m_derivationFunction(neuron->output);
							neuron->delta = error;
						}
					}
				}
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
						n->updateWeights(m_layers[i - 1], m_derivationFunction);
					}
				}
			}
		}
		template<typename T>
		std::vector<T> classify(sANN::dataset<T> &set)
		{
			if (set.training.size() != 0 || !initialized)
			{
				std::runtime_error("Network not initialized or trainingdata included");
				std::exit(-1);
			}
			std::vector<T> ret;
			for (auto vec : set.data)
			{
				layer l = this->feedForward(vec);
				for (auto val : l)
				{
					ret.push_back(val->output);
				}
			}

			return ret;
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

			auto layers = sANN::internals::split(str, '\n');
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
	};
}
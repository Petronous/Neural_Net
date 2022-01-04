#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "utils.hpp"
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>


// ADD PROPERTY FOR CHANCE OF MUTATION
class NeuralNet
{
private:
	std::size_t layers_num{ 0 };
	std::vector<MyMat> layers_weights;
	float _mutate_mod_range;
	float _mutate_rand_range;
	void set_weight(int layer, int from_neuron, int to_neuron, float weight) {
		layers_weights[layer](to_neuron, from_neuron) = weight;
	}
	void add_to_weight(int layer, int from_neuron, int to_neuron, float weight) {
		layers_weights[layer](to_neuron, from_neuron) += weight;
	}

public:
	float _mutate_mod_prob;
	float _mutate_rand_prob; //should be read-only
	float score = 0;

	/**
	* Makes randomly from layer sizes
	*/
	NeuralNet(std::vector<int> &layers, float mutate_mod_prob=0.2, float mutate_mod_range=0.1, float mutate_rand_prob=0.1, float mutate_rand_range=2.):
		layers_num(layers.size()-1), //one layer isn't neurons but instead is input
		_mutate_mod_prob(mutate_mod_prob),
		_mutate_mod_range(mutate_mod_range),
		_mutate_rand_prob(mutate_rand_prob),
		_mutate_rand_range(mutate_rand_range)
	{
		int last_layer_size = 0;
		layers_weights.reserve(layers.size());
		for (auto layer_size : layers) {
			if (last_layer_size != 0) {
				layers_weights.push_back(MyMat(last_layer_size, layer_size).setRandom());
			}
			last_layer_size = layer_size;
		}
	}

	/**
	* Makes exactly from specified weights
	*/
	NeuralNet(std::vector<MyMat> &weights, float mutate_mod_prob=0.2, float mutate_mod_range=0.1, float mutate_rand_prob=0.1, float mutate_rand_range=2.):
		layers_num(weights.size()),
		layers_weights(weights),
		_mutate_mod_prob(mutate_mod_prob),
		_mutate_mod_range(mutate_mod_range),
		_mutate_rand_prob(mutate_rand_prob),
		_mutate_rand_range(mutate_rand_range)
	{}

	/**
	*
	* Loads from string
	* 
	*/
	NeuralNet(std::string line) {
		std::istringstream input{ line };
		DBOUT("\"" + line + "\"");
		assert(line != "");
		std::string token = "";
		int layer_size;
		int last_layer_size = -1;
		input >> token;
		layers_num = 0;

		while (token != "|") {
			layer_size = std::stoi(token);
			if (last_layer_size != -1) {
				layers_weights.push_back(MyMat(last_layer_size, layer_size));
				layers_num++;
			}
			last_layer_size = layer_size;

			input >> token;
		}
		
		input >> _mutate_mod_prob;
		input >> _mutate_mod_range;
		input >> _mutate_rand_prob;
		input >> _mutate_rand_range;

		std::vector<int> architecture = get_layer_architecture();
		int x_size;
		int y_size;
		for (int i = 0; i < layers_num; i++) {
			x_size = architecture[i];
			y_size = architecture[i + 1];
			for (int x = 0; x < x_size; x++) {
				for (int y = 0; y < y_size; y++) {
					input >> layers_weights[i](x, y);
				}
			}
		}
	}

	/**
	* @brief input will be modified by this function and also serve to hold the result
	* 
	* input will be modified by this function and also serve to hold the result.
	* DO NOT forget that the input is squashed first.
	*/
	void step(MyMat &input) {
		input = input.unaryExpr([](float x) { return std::tanh(x); });
		for (int i = 0; i < layers_num; i++) {
			//DBOUT(my_to_string(input));
			input = (input * layers_weights[i]).unaryExpr([](float x) { return std::tanh(x); });
		}
	}


	void mutate_random_weight(bool set_random = false, int precision = 100000) {
		int layer_i = std::rand() % layers_num;
		MyMat& layer = layers_weights[layer_i];
		int from = std::rand() % layer.cols();
		int to = std::rand() % layer.rows();
		if(set_random) {
			float weight = rand_float_in_range(-_mutate_rand_range, _mutate_rand_range, precision);
			set_weight(layer_i, from, to, weight);
		} else {
			float weight = rand_float_in_range(-_mutate_mod_range, _mutate_mod_range, precision);
			add_to_weight(layer_i, from, to, weight);
		}
	}

	void set_score(float x) {
		score = x;
	}
	float get_score() {
		return score;
	}

	void set_mutation_params(float mod_prob = -1, float mod_range = -1, float rand_prob = -1, float rand_range = -1) {
		if(mod_prob != -1) _mutate_mod_prob = mod_prob;
		if(mod_range != -1) _mutate_mod_range = mod_range;
		if(rand_prob != -1) _mutate_rand_prob = rand_prob;
		if (rand_range != -1) _mutate_rand_range = rand_range;
	}

	void mutate_mutation_param(bool set_random = false, int precision = 100000) {
		int param_id = std::rand() % 4;
		float* param = &_mutate_mod_prob;
		float mn = 1. / precision;
		float mx = 1. - 1. / precision;
		if (param_id == 1) {
			param = &_mutate_mod_range;
			mn = 0;
			mx = 1;
		}
		if (param_id == 2) param = &_mutate_rand_prob;
		if (param_id == 3) {
			param = &_mutate_rand_range;
			mn = 0;
			mx = 10;
		}

		if (set_random) {*param = rand_float_in_range(mn, mx, precision);} 
		else {*param = clamp(*param + rand_float_in_range(-_mutate_mod_range/10, _mutate_mod_range/10, precision), mn, mx);}

	}

	std::vector<int> get_layer_architecture() {
		std::vector<int> layers;
		layers.reserve(layers_num+1);
		layers.push_back(layers_weights[0].rows());
		for (int i = 0; i < layers_num; i++){
			layers.push_back(layers_weights[i].cols());
		}
		return layers;
	}

	std::ostream& operator<<(std::ostream& os) {
		os << to_string();
		return os;
	}

	std::string to_string(bool print_net=false) {
		std::ostringstream return_stream;
		return_stream << layers_num << "\n";
		return_stream << "MMP: " << _mutate_mod_prob << " MMR: " << _mutate_mod_range << " MRP: " << _mutate_rand_prob << " MRR: " << _mutate_rand_range << "\n";
		if (print_net) {
			for (int i = 0; i < layers_num; i++) {
				return_stream << "---\n" << layers_weights[i] << "\n";
			}
		}
		return return_stream.str();
	}

	void save_to_file(std::ofstream& output) {
		std::vector<int> architecture = get_layer_architecture();
		for (auto i : architecture) {
			output << i << " ";
		}
		output << " | ";
		output << _mutate_mod_prob << " " << _mutate_mod_range << " " << _mutate_rand_prob << " " << _mutate_rand_range << " ";
		int x_size;
		int y_size;
		for (int i = 0; i < layers_num; i++) {
			x_size = architecture[i];
			y_size = architecture[i+1];
			for (int x = 0; x < x_size; x++) {
				for (int y = 0; y < y_size; y++) {
					output << layers_weights[i](x, y) << " ";
				}
			}
		}
		output << "\n";
	}
};


bool compare_scores(NeuralNet n, NeuralNet m) {
	return n.score > m.score;
}

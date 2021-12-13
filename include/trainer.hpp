#pragma once

#include <time.h>

class Trainer {
	std::vector<NeuralNet> _population;
	int _pop_size;
	float _mutate_mod_prob;
	float _mutate_rand_prob;
	int _mutate_prob_precision;
	bool _force_mutation;
	bool _mass_eval_bool;
	float _kill_frac;
	void (*_single_eval)(NeuralNet&);
	void (*_mass_eval)(std::vector<NeuralNet>&);

	NeuralNet get_best_net() {
		float max_score = _population[0].get_score();
		int max_i = 0;
		float curr_score;
		for (int i = 1; i < _pop_size; i++) {
			curr_score = _population[i].get_score();
			if (max_score < curr_score) {
				max_score = curr_score;
				max_i = i;
			}
		}
		return _population[max_i];
	}

	void sort_population() {
		std::sort(_population.begin(), _population.end(), compare_scores);
	}

	void reproduce(int from, int to) {
		_population[to] = _population[from];
		NeuralNet& current = _population[to];
		if (_force_mutation) {
			current.mutate_random_weight();
		}
		int mutation_counter = 0;
		while (current._mutate_rand_prob < rand_float_in_range(0, 1, _mutate_prob_precision && mutation_counter < 100)) {
			current.mutate_random_weight(true);
			mutation_counter++;
		}
		mutation_counter = 0;
		while (current._mutate_mod_prob < rand_float_in_range(0, 1, _mutate_prob_precision) && mutation_counter < 100) {
			current.mutate_random_weight();
			mutation_counter++;
		}

		mutation_counter = 0;
		while (_mutate_rand_prob / 10 < rand_float_in_range(0, 1, _mutate_prob_precision) && mutation_counter < 10) {
			current.mutate_mutation_param(true);
			mutation_counter++;
		}
		mutation_counter = 0;
		while (_mutate_mod_prob / 10 < rand_float_in_range(0, 1, _mutate_prob_precision) && mutation_counter < 10) {
			current.mutate_mutation_param();
			mutation_counter++;
		}
	}

	void reproduce_fair(float& max_score, NeuralNet& best) {
		sort_population();
		max_score = _population[0].score;
		best = _population[0];
		int first_death_i = max(_pop_size * (1-_kill_frac), 1);
		int copy_from = 0;
		for (int i = first_death_i; i < _pop_size; i++) {
			reproduce(copy_from, i);
			//DBOUT(my_to_string(copy_from) + " " + my_to_string(i));

			copy_from = (copy_from + 1) % first_death_i;
		}
	}

	void reproduce_best(int& max_score, NeuralNet& best) {
		best = get_best_net();
		max_score = best.score;
		_population[0] = best;
		for (int i = 1; i < _pop_size; i++) {
			reproduce(0, i);
		}
	}

public:
	/**
	* @brief Requires set_eval_func() to be called as well, won't work otherwise
	*/
	Trainer(NeuralNet starting_net, int pop_size = 50, float mutate_mod_prob = 0.02,
		float mutate_rand_prob = 0.004, float kill_frac = 0.5, int mutate_prob_precision = 1000,
		bool force_mutation = true, bool randomize = true):

		_mutate_mod_prob(mutate_mod_prob),
		_mutate_rand_prob(mutate_rand_prob),
		_mutate_prob_precision(mutate_prob_precision),
		_force_mutation(force_mutation),
		_mass_eval_bool(false),
		_pop_size(pop_size),
		_kill_frac(kill_frac)
	{
		_population.reserve(pop_size);
		_population.push_back(starting_net);
		std::vector<int> architecture = starting_net.get_layer_architecture();
		for (int i = 1; i < pop_size; i++) {
			if (randomize) {
				_population.push_back(NeuralNet(architecture, mutate_mod_prob, 0.1, mutate_rand_prob));
			}
			else {
				starting_net.set_score(0);
				_population.push_back(starting_net);
				_population[i].set_mutation_params(mutate_mod_prob, 0.1, mutate_rand_prob, 2);
				_population[i].mutate_random_weight();
			}
		}
	}

	void set_eval_func(void (*single_eval)(NeuralNet&)) {
		_single_eval = single_eval;
		_mass_eval_bool = false;
	}
	void set_eval_func(void (*mass_eval)(std::vector<NeuralNet>&)) {
		_mass_eval = mass_eval;
		_mass_eval_bool = true;
	}

	NeuralNet train(int max_gen = 100, float stop_after_score = FLT_MAX, int max_time = 1 << 30) {
		float max_score = -FLT_MAX;
		int generation = 0;
		int start_time = std::time(NULL);
		NeuralNet best = get_best_net();

		//don't worry, there's a breaking if a little later
		while (true) {
			if (_mass_eval_bool) {
				_mass_eval(_population);
			} else {
				for (int i = 0; i < _pop_size; i++) {
					srand(RANDOM_SEED+generation);
					_single_eval(_population[i]);
				}
			}

			if (!(generation < max_gen && max_score < stop_after_score && std::time(NULL) - start_time < max_time)) break;
			reproduce_fair(max_score, best);
			
			DBOUT(my_to_string(generation) + " " + my_to_string(max_score) + "\n***\n" + best.to_string());
			generation++;
		}
		sort_population();
		best = _population[0];

		return best;
	}

	void print_pop_scores() {
		std::ostringstream oss;
		for (int i = 0; i < _pop_size; i++) {
			oss << my_to_string(_population[i].score);
			oss << "\n";
		}
		DBOUT(oss.str());
	}
};

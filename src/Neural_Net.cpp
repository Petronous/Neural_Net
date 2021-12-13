// Neural_Net.cpp : Defines the entry point for the application.
//

#include "Neural_Net.h"


void evaluate_one(NeuralNet& net) {
	int auto_inputs = 0;
	float a, b;
	MyMat input(4 + auto_inputs, 3);
	input.block(0,0,4,3) << 10., 10., 10.,
		    -10., 10., 10.,
		     10.,-10., 10.,
		    -10.,-10., 10.
		;
	MyMat ideal(4 + auto_inputs, 1);
	ideal.block(0,0,4,1) << -1, 1, 1, -1;

	for (int i = 0; i < auto_inputs; i++) {
		a = rand_float_in_range(-10, -0.5, 10000);
		if (rand() % 2) a *= -1;
		b = rand_float_in_range(-10, -0.5, 10000);
		if (rand() % 2) b *= -1;
		input.row(i+4) << a, b, 10.;
		if (!(a > 0) != !(b > 0)) ideal.row(i+4) << 1;
		else ideal.row(i+4) << -1;
	}
	//DBOUT(my_to_string(input));
	//DBOUT(my_to_string(ideal));
	net.step(input);
	//DBOUT(my_to_string(input));
	net.score = -10 * (input.block(0, 0, 4, 1) - ideal.block(0, 0, 4, 1)).cwiseAbs().sum();
	net.score -= -(input.block(4, 0, auto_inputs, 1) - ideal.block(4, 0, auto_inputs, 1)).cwiseAbs().sum();
}


int main() 
{
	std:srand(RANDOM_SEED);
	MyMat l1(3,3);
	l1 << -10, 10, 0,
		  -10, 10, 0,
		   10, 10, 10
		;
	MyMat l2(3,1);
	l2 << 2, 
		  2,
	     -1;
	MyMat input(4,3);
	input << 10., 10., 10.,
		    -10., 10., 10.,
		     10.,-10., 10.,
		    -10.,-10., 10.
		;
	std::vector<MyMat> weights = {l1, l2};
	NeuralNet net{weights};
	net.step(input);
	DBOUT(my_to_string(input));
	NeuralNet net2(std::vector<int>{3, 3, 1});
	net2.mutate_mutation_param();
	DBOUT(net2.to_string());
	net2.mutate_mutation_param();
	DBOUT(net2.to_string());
	net2.mutate_mutation_param();
	DBOUT(net2.to_string());
	net2.mutate_mutation_param(true);
	DBOUT(net2.to_string());
	DBOUT(net.to_string());
	DBOUT(my_to_string(net.get_layer_architecture()));

	std::ofstream output_file{ "best_in_gen.txt", std::ios::app };
	net2.save_to_file(output_file);
	output_file.close();


	NeuralNet start_net(std::vector<int>{3, 4, 4, 1});
	Trainer trainer(start_net, 100, 0.4, 0.2, 0.5);
	trainer.set_eval_func(&evaluate_one);

	start_net = trainer.train(100);
	
	DBOUT(start_net.to_string());
	DBOUT(my_to_string(start_net.score));
	input = MyMat(4, 3);
	input << 10., 10., 10.,
		    -10., 10., 10.,
		     10.,-10., 10.,
		    -10.,-10., 10.
		;
	start_net.step(input);
	DBOUT(my_to_string(input));
	//trainer.print_pop_scores();
}

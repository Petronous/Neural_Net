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
	MyMat input_copy = input;
	std::vector<MyMat> weights = {l1, l2};
	NeuralNet net{weights};
	net.step(input_copy);
	DBOUT("################\nNET SAVING TEST\n################");
	DBOUT(my_to_string(input_copy));
	NeuralNet net2(std::vector<int>{3, 3, 1});
	/*net2.mutate_mutation_param();
	DBOUT(net2.to_string());
	net2.mutate_mutation_param();
	DBOUT(net2.to_string());
	net2.mutate_mutation_param();
	DBOUT(net2.to_string());
	net2.mutate_mutation_param(true);
	DBOUT(net2.to_string());
	DBOUT(net.to_string());
	DBOUT(my_to_string(net.get_layer_architecture()));*/

	std::ofstream output_file{ "best_in_gen.txt", std::ios::app };
	net.save_to_file(output_file);
	output_file.close();
	
	std::ifstream input_file{ "best_in_gen.txt" };
	if (!input_file) DBOUT("File doesn't exist");
	std::string line;
	std::getline(input_file, line);
	net = NeuralNet(line);
	net.step(input);
	DBOUT(my_to_string(input));
	assert(input == input_copy);	

	std::ofstream training_output_file{ "best_in_gen.txt", std::ios::app };
	NeuralNet start_net(std::vector<int>{3, 4, 4, 1});
	start_net = NeuralNet("3 4 4 1  | 0.280022 0.267574 0.0264995 1.93151 -0.768142 -1.37928 -0.378894 0.781305 -1.04062 0.349846 0.242793 -1.22657 -1.57911 -1.00262 -0.144153 -0.886128 0.429774 0.641107 0.909604 -0.0583819 -0.867243 -0.478222 -0.911111 -0.766711 -1.14232 -0.0249907 -0.444647 -0.471706 0.632796 -1.23267 -1.19882 -0.255469 -1.55873 -3.19319 -3.0206 -1.32395 ");
	Trainer trainer(start_net, 200, 0.4, 0.2, 0.5);
	trainer.set_eval_func(&evaluate_one);

	start_net = trainer.train(training_output_file, 100);
	
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

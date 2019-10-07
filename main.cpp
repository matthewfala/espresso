#include <vector>
#include <iostream>
#include <limits>

#include "Tensor.h"

using namespace std;
int main()
{
	std::cerr << "Tensor Test Program" << std::endl;

	std::cerr <<  "Zero Matix"
	Tensor t();
	t.ZeroInit(5, 5);
	
	std::cerr << "Tensor: " << t.getRows() << std::endl;
	std::cerr << "Tensor: " << t.getCols() << std::endl;
	std::cerr << "Tensor 4, 4" << t.at(4, 4) << " ." << std::endl;

	std::cerr << "Print Matrix" << std::endl;
	t.print();


	std::cerr << "Random Matix"
	Tensor b();
	b.RandInit(5, 5);

	std::cerr << "Tensor: " << b.getRows() << std::endl;
	std::cerr << "Tensor: " << b.getCols() << std::endl;

	std::cerr << "Tensor: " << b.getRows() << std::endl;
	std::cerr << "Tensor: " << b.getCols() << std::endl;
	std::cerr << "Tensor 4, 4" << b.at(4, 4) << " ." << std::endl;
	b.print();


	return 0;
}


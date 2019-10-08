#include <vector>
#include <iostream>
#include <limits>

#include "Espresso/Tensor.h"
using std::vector;

int main()
{
	std::cerr << std::endl << "Tensor Test Program" << std::endl;

	std::cerr << "Zero Matix" << std::endl;
	Tensor t;
	std::cerr << "ColCount " << t.GetCols() << std::endl;
	t.ZeroInit(5, 5);
	
	std::cerr << "Rows: " << t.GetRows() << std::endl;
	std::cerr << "Cols: " << t.GetCols() << std::endl;
	std::cerr << "Val at 4, 4: " << t.at(4, 4) << " ." << std::endl;

	std::cerr << std::endl << "Print Matrix" << std::endl;
	
	//t.Print();

	vector<vector<float>> x {

		{1, 7, 3},
		{4, 5, 6}
	};

	Tensor b(x);

	b.Print();

	std::cerr << std::endl <<  "Translate Matrix" << std::endl;

	b.Translate();
	b.Print();

	std::cerr << std::endl << "Single Value" << std::endl;
	Tensor c(10.0f);

	c.Print();

	
	std::cerr << std::endl << "Copy Constructor" << std::endl;
	Tensor d(b);
	d.Print();
	d.at(0, 0) = 100.0f;
	d.Translate();
	d.Print();
	b.Print();

	std::cerr << std::endl << "Destructor Stress" << std::endl;

	vector<Tensor*> tensors;
	std::cerr << "Pre Allocation breakpoint" << std::endl;

	for (int i = 0; i < 10000; i++) {
		tensors.emplace_back(new Tensor(b));
	}

	std::cerr << "Allocated 10000 tensors" << std::endl;

	for (auto t : tensors) {
		delete t;
	}

	std::cerr << "Deallocated Memory";

	std::cerr << std::endl << "Operator Test" << std::endl;
	Tensor e(t);
	e.Print();
	e = b;
	e.Print();
	Tensor f;
	f = b;
	f.Translate();

	f.Print();
	Tensor g = e.Dot(f);

	g.Print();

	std::cerr << std::endl << "Add 100, subtract 100, mult 5, div 10" << std::endl;
	g += 100;
	g.Print();
	g -= 100;
	g.Print();
	g *= 5;
	g.Print();
	g /= 10;
	g.Print();


	std::cerr << std::endl << "Max Check" << std::endl;
	Tensor h = g.Reduce(0, [] (float acc, float cur) {
		return (cur > acc) ? cur : acc;
		});

	h.Print();

	Tensor i = h.Max(0).Max(1);
	i.Print();



	/*
	std::cerr << "Random Matix" << std::endl;
	Tensor b();
	b.RandInit(5, 5);

	std::cerr << "Tensor: " << b.getRows() << std::endl;
	std::cerr << "Tensor: " << b.getCols() << std::endl;

	std::cerr << "Tensor: " << b.getRows() << std::endl;
	std::cerr << "Tensor: " << b.getCols() << std::endl;
	std::cerr << "Tensor 4, 4" << b.at(4, 4) << " ." << std::endl;
	b.print();
	*/

	return 0;
}


#include <vector>
#include <iostream>
#include <limits>

#include "Espresso/Tensor.h"
#include "Espresso/DataBatcher.hpp"
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

	vector<vector<float>> x{

		{1, 7, 3},
		{4, 5, 6}
	};

	Tensor b(x);

	b.Print();

	std::cerr << std::endl << "Transpose Matrix" << std::endl;

	b.Transpose();
	b.Print();

	std::cerr << std::endl << "Single Value" << std::endl;
	Tensor c(10.0f);

	c.Print();


	std::cerr << std::endl << "Copy Constructor" << std::endl;
	Tensor d(b);
	d.Print();
	d.at(0, 0) = 100.0f;
	d.Transpose();
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
	f.Transpose();

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
	Tensor h = b.Reduce(0, [](float acc, float cur) {
		return (cur > acc) ? cur : acc;
		});

	h.Print();

	Tensor i = b.Max(0).Max(1);
	i.Print();

	Tensor::Seed(5);

	Tensor j1, j2, j3;
	j1.RandInit(3, 4, -.5, .5);
	j2.RandInit(3, 4, -.5, .5);
	j3.RandInit(3, 4, -.5, .5);

	j1.Print();
	j2.Print();
	j3.Print();


	std::cerr << "Subtract Column Tensor" << std::endl;
	Tensor k(b);
	Tensor l(vector<vector<float>>{ {-40, -10 }  } );

	k.Print();
	l.Print();
	k -= l;
	k.Print();

	std::cerr << "Subtract Row Tensor" << std::endl;

	k = b;
	k.Print();
	Tensor m = Tensor(vector<vector<float>>{ {-40}, { -10 }, { -30 } });
	m.Print();
	k -= m;
	k.Print();

	// Equality operator
	std::cerr << "Assignment op" << std::endl;

	k.Print();
	d.Print();

	k = d;
	k.Print();

	std::cerr << "Dot Test" << std::endl;
	Tensor o = b;
	vector<vector<float>> y{

	{1, 7},
	{2, 5}
	};

	
	Tensor p(y);
	p += 5;
	p.Transpose();

	o.Print();
	p.Print();
	Tensor q = o.DotT(p);
	q.Print();

	
	std::cerr << "Rand Dot" << std::endl;

	Tensor weight;
	weight.RandInit(101, 784, -.5, .5);

	Tensor myX; 
	myX.Transpose();
	myX.RandInit(784, 4, -.5, .5);

	Tensor myDot;
	myDot = weight.DotT(myX);
	myDot.Print();



	





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


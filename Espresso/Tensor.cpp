#include "Tensor.h"

// constructors
Tensor::Tensor(float init) {
	std::vector<std::vector<float>> t{ {init} };
	AllocTensor(1, 1);
	InitTensor(t);
}

Tensor::Tensor(std::vector<std::vector<float>> init);
Tensor::Tensor(std::vector<std::vector<int>> init);

void Tensor::AllocTensor(size_t rows, size_t cols) {


	size_t rows = t.size();
	size_t rows = t.size();

	SetRows(rows);
	SetCols(cols);

	data = new float*[rows];
	for (int j = 0; j < cols; ++j) {
		data[j] = new float[cols];
	}

	
	data = new float[rows, cols];

}

void Tensor::DeallocTensor();

void Tensor::InitTensor(std::vector<std::vector<float>> t) {


	// set matrix
	for (size_t i = 0; i < mRows; ++i) {
		for (size_t j = 0; j < mCol; ++j) {
			at(i, j) = t[i][j];
		}
	}
}


// destructor
Tensor::~Tensor();

// copy constructor
Tensor::Tensor(const Tensor& rhs);

// init
void ZeroInit(size_t r, size_t col);
void RandInit(size_t r, size_t col);



float& at(size_t i, size_t j);
std::vector<float> at(size_t i);
void translate();

// math
void dot(Tensor o);
Num& operator+=(const Num& rhs);
Num& operator-=(const Num& rhs);
Num& operator*=(float rhs);
Num& operator/=(float rhs);
vector<float> max(size_t direction);

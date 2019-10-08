#include "Tensor.h"
#include <iostream>
#include <utility>

// Constructors
Tensor::Tensor(float init) {
	std::vector<std::vector<float>> t{ {init} };
	AllocTensor(1, 1);
	SetData(t);
}


Tensor::Tensor(const std::vector<std::vector<float>>& init) {
	AllocTensor(init.size(), init[0].size());
	SetData(init);
}


Tensor::Tensor(const std::vector<std::vector<int>>& init) {
	// convert to float vector
	std::vector<std::vector<float>> conv;

	for (size_t i = 0; i < init.size(); ++i) {
		conv.emplace_back(std::vector<float>{});
		for (size_t j = 0; j < init[0].size(); ++j) {
			conv[i].emplace_back(static_cast<float>(init[i][j]));
		}
	}

	AllocTensor(init.size(), init[0].size());
	SetData(conv);

}

// Move Constructor 
Tensor::Tensor(Tensor&& rhs)
	: mIsTranslated(std::move(rhs.mIsTranslated))
	, mRows(std::move(rhs.mRows))
	, mCols(std::move(rhs.mCols))
	, mData(std::move(rhs.mData))
{
	rhs.mData = nullptr;
	rhs.mRows = 0;
	rhs.mCols = 0;
}

// Destructor
Tensor::~Tensor() {
	DeallocTensor();
}

// copy constructor
Tensor::Tensor(const Tensor& rhs)
	: mIsTranslated(rhs.mIsTranslated)
	, mRows(rhs.mRows)
	, mCols(rhs.mCols)
{
	mData = new float* [mRows];
	for (int i = 0; i < mRows; ++i) {
		mData[i] = new float[mCols];
		for (int j = 0; j < mCols; ++j) {
			mData[i][j] = rhs.mData[i][j];
		}
	}
}

// equality operator (Does this also count as a move constructor??????) - Ask Prof Sanjay
Tensor& Tensor::operator=(Tensor rhs)
{
	// utilize swap trick
	using std::swap;
	swap(mIsTranslated, rhs.mIsTranslated);
	swap(mRows, rhs.mRows);
	swap(mCols, rhs.mCols);
	swap(mData, rhs.mData);

	return *this;
}

// Memory Management
void Tensor::AllocTensor(size_t rows, size_t cols) {

	// ensure memory is cleared;
	DeallocTensor();

	Rows() = rows;
	Cols() = cols;

	std::cerr << "Rows: " << Rows() << std::endl;
	std::cerr << "Cols: " << Cols() << std::endl;
	std::cerr << "mRows: " << mRows << std::endl;
	std::cerr << "mCols: " << mCols << std::endl;

	// Create based on data specs mRows and mCols
	mData = new float* [mRows];
	for (int i = 0; i < mRows; ++i) {
		mData[i] = new float[mCols];
	}

}


void Tensor::DeallocTensor() {
	for (int i = 0; i < mRows; ++i) {
		delete[] mData[i];
	}
	delete[] mData;

	mCols = 0;
	mRows = 0;
	mData = nullptr;
}


void Tensor::SetData(const std::vector<std::vector<float>>& t) {

	// check
	if (t.size() != GetRows() || t[0].size() != GetCols()) {
		std::cerr << "Shape Error: Set data passed in an incorrect shape" << std::endl;
		return;
	}

	std::cerr << "Setting with Vector Shape: " << t.size() << ", " << t[0].size() << std::endl;
	std::cerr << "Matrix Shape             : " << GetRows() << ", " << GetCols() << std::endl;
	////std::cerr << "Matrix Shape             : " << GetRows() << ", " << GetCols() << std::endl;


	// set matrix
	for (size_t i = 0; i < Rows(); ++i) {
		for (size_t j = 0; j < Cols(); ++j) {
			at(i, j) = t[i][j];
		}
	}

}

// Initialization
void Tensor::ZeroInit(size_t r, size_t c) {

	// create matrix
	std::vector<std::vector<float>> z;
	for (int i = 0; i < r; ++i) {
		z.emplace_back(std::vector<float>{});
		for (int j = 0; j < c; ++j) {
			z[i].emplace_back(0.0f);
		}
	}

	// alloc matrix and set data
	AllocTensor(r, c);
	SetData(z);
}

void Tensor::RandInit(size_t r, size_t c) {
	return;
}


// Utility
void Tensor::Print() {
	std::cout << Rows() << " x " << Cols() << " Matrix. " << (mIsTranslated ? "Translated" : "Not Translated") << std::endl;
	std::cout << "[";
	for (int i = 0; i < Rows(); ++i) {
		std::cout << "[	";
		for (int j = 0; j < Cols(); ++j) {
			std::cout << at(i, j) << ",	";
		}
		std::cout << "	]" << std::endl;
	}
	std::cout << "]	" << std::endl;

}

// math
void Tensor::ForEach(std::function<void(float&)> f) {
	for (size_t i = 0; i < mRows; ++i) {
		for (size_t j = 0; j < mCols; ++j) {
			f(mData[i][j]);
		}
	}
}

Tensor Tensor::Reduce(size_t axis, std::function<float(float, float)> reducer) {

	// Zero init acc & collect step data
	Tensor acc;
	std::pair<size_t, size_t> acrossStep(0, 0);
	std::pair<size_t, size_t> reduceStep(0, 0);
	size_t reductions = 0;
	size_t crosses = 0;

	if (axis == 0) {
		acc.ZeroInit(1, GetCols());
		acrossStep.second = 1;
		reduceStep.first = 1;
		crosses = GetCols();
		reductions = GetRows();
	}
	else if (axis == 1) {
		acc.ZeroInit(GetRows(), 1);
		acrossStep.first = 1;
		reduceStep.second = 1;
		crosses = GetRows();
		reductions = GetCols();
	}
	else {
		std::cerr << "Error: Reduce supplied out of bounds axis. Only [0, 1] supported" << std::endl;
		return *this;
	}

	// fill the accumulator
	size_t i = 0;
	size_t j = 0;
	for (size_t r = 0; r < reductions; ++r) {
		
		i -= i * acrossStep.first;
		j -= j * acrossStep.second;
		for (size_t c = 0; c < crosses; ++c) {
			size_t redI = acrossStep.first * i;
			size_t redJ = acrossStep.second * j;
			acc.at(redI, redJ) = reducer(acc.at(redI, redJ), at(i, j));
			i += acrossStep.first;
			j += acrossStep.second;
		}
		
		i += reduceStep.first;
		j += reduceStep.second;
	}

	// return the accumulator
	return acc;
}

Tensor Tensor::Dot(Tensor o) { 

	// sentinal
	if (GetCols() != o.GetRows()) {
		std::cout << "Error: Dot product dimensions are incompatible" << std::endl;
		return *this;
	}

	// dot
	Tensor d;
	d.ZeroInit(GetRows(), o.GetCols());

	for (int i = 0; i < GetRows(); ++i) {
		for (int j = 0; j < o.GetCols(); ++j) {
			for (int k = 0; k < GetCols(); ++k) {
				d.at(i, j) += at(i, k) * o.at(k, j);
			}
		}
	}

	return d;
}

Tensor& Tensor::operator+=(const Tensor& rhs) { 

	// sentinal
	if (GetCols() != rhs.GetCols() || GetRows() != rhs.GetRows()) {
		std::cout << "Error: Addition shapes do not match" << std::endl;
		return *this;
	}

	// copy the data
	for (size_t i = 0; i < GetRows(); ++i) {
		for (size_t j = 0; j < GetCols(); ++j) {
			at(i, j) += rhs.atC(i, j);
		}
	}
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
	// sentinal
	if (GetCols() != rhs.GetCols() || GetRows() != rhs.GetRows()) {
		std::cout << "Error: Subtraction shapes do not match" << std::endl;
		return *this;
	}

	// copy the data
	for (size_t i = 0; i < GetRows(); ++i) {
		for (size_t j = 0; j < GetCols(); ++j) {
			at(i, j) -= rhs.atC(i, j);
		}
	}
}

// Scalar Operators
Tensor& Tensor::operator+=(float rhs) {
	ForEach([rhs](float& d) { d += rhs; });
	return *this;
}

Tensor& Tensor::operator-=(float rhs) {
	ForEach([rhs](float& d) { d -= rhs; });
	return *this;
}

Tensor& Tensor::operator*=(float rhs) { 
	ForEach([rhs](float& d) { d *= rhs; });
	return *this;
}

Tensor& Tensor::operator/=(float rhs) { 
	ForEach([rhs](float& d) { d /= rhs; });
	return *this;
}

Tensor Tensor::Max(size_t direction) {
	return Reduce(direction, [](float acc, float cur) {
		return (cur > acc) ? cur : acc;
		});
}

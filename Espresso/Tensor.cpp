#include "Tensor.h"
#include <utility>
#include <random>

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
Tensor::Tensor(Tensor&& rhs) noexcept
	: mIsTransposed(std::move(rhs.mIsTransposed))
	, mToggleOffBias(std::move(rhs.mToggleOffBias))
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
	: mIsTransposed(rhs.mIsTransposed)
	, mToggleOffBias(rhs.mToggleOffBias)
	, mRows(rhs.mRows)
	, mCols(rhs.mCols)
{

	if (this == &rhs) {
		return;
	}

	mData = new float* [mRows];
	for (int i = 0; i < mRows; ++i) {
		mData[i] = new float[mCols];
		for (int j = 0; j < mCols; ++j) {
			mData[i][j] = rhs.mData[i][j];
		}
	}
}

// equality operator (Does this also count as a move constructor??????) - Ask Prof Sanjay
Tensor& Tensor::operator=(Tensor& rhs)
{
	// sentinel
	if (&rhs == this) {
		return *this;
	}

	// remove data
	DeallocTensor();

	// copy the data
	mIsTransposed = rhs.mIsTransposed;
	mToggleOffBias = rhs.mToggleOffBias;
	mRows = rhs.mRows;
	mCols = rhs.mCols;

	// alloc and fill
	mData = new float* [mRows];
	for (int i = 0; i < mRows; ++i) {
		mData[i] = new float[mCols];
		for (int j = 0; j < mCols; ++j) {
			mData[i][j] = rhs.mData[i][j];
		}
	}

	return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& rhs) noexcept {
	if (&rhs == this) {
		return *this;
	}

	mIsTransposed = std::move(rhs.mIsTransposed);
	mToggleOffBias = std::move(rhs.mToggleOffBias);
	mRows = std::move(rhs.mRows);
	mCols = std::move(rhs.mCols);
	mData = std::move(rhs.mData);

	// don't forget to wipe rhs
	rhs.mData = nullptr;
	rhs.mRows = 0;
	rhs.mCols = 0;

	return *this;
}


// Memory Management
void Tensor::AllocTensor(size_t rows, size_t cols) {

	// ensure memory is cleared;
	DeallocTensor();

	Rows() = rows;
	Cols() = cols;

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

	// Helpful debug shape printing
	// std::cerr << "Allocating Tensor of Shape: " << t.size() << ", " << t[0].size() << std::endl;
	// std::cerr << "Matrix Shape             : " << GetRows() << ", " << GetCols() << std::endl;
	////std::cerr << "Matrix Shape             : " << GetRows() << ", " << GetCols() << std::endl;


	// set matrix
	for (size_t i = 0; i < Rows(); ++i) {
		for (size_t j = 0; j < Cols(); ++j) {
			at(i, j) = t[i][j];
		}
	}

}


// Extract data as a vector<vector<float>>
std::vector<std::vector<float>> Tensor::GetData() const {
	std::vector<std::vector<float>> dat;

	for (int i = 0; i < GetRows(); ++i) {

		dat.emplace_back(std::vector<float>());
		for (int j = 0; j < GetCols(); ++j) {
			dat[i].emplace_back(atC(i, j));
		}
	}
	return dat;

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

unsigned int Tensor::mSeed = 0;

void Tensor::RandInit(size_t r, size_t c, float min, float max, unsigned int seed) {


	std::mt19937 generator(mSeed);  // mt19937 is a standard mersenne_twister_engine
	std::cout << "Random value: " << generator() << std::endl;
	++mSeed;
	
	// create matrix
	std::vector<std::vector<float>> v;
	for (size_t i = 0; i < r; ++i) {
		v.emplace_back(std::vector<float>{});
		for (size_t j = 0; j < c; ++j) {
			v[i].emplace_back( (max - min) * (static_cast<float>(generator()) / (generator.max())) + min );
		}
	}

	// alloc matrix and set data
	AllocTensor(r, c);
	SetData(v);

}


// Utility
void Tensor::Print() {
	std::cout << GetRows() << " x " << GetCols() << " Matrix. " << (mIsTransposed ? "Transposed" : "Not Transposed") << std::endl;
	std::cout << "[";
	for (int i = 0; i < GetRows(); ++i) {
		std::cout << "[	";
		for (int j = 0; j < GetCols(); ++j) {
			std::cout << at(i, j) << ",	";
		}
		std::cout << "	]" << std::endl;
	}
	std::cout << "]	" << std::endl;

}

void Tensor::PrintShape() {
	std::cout << GetRows() << " x " << Cols() << " Matrix. " << (mIsTransposed ? "Transposed" : "Not Transposed") << std::endl;
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


Tensor& Tensor::TwoTensorOp(std::function<float(float, float)> op, const Tensor& rhs) {

	// Special: Row wise subtraction
	size_t iStep = 0;
	size_t jStep = 0;
	size_t iNext = 0;
	size_t jNext = 0;
	size_t stepCount, nextCount;

	bool lineWise = false;

	// this -> (x) rhs -> (|) ( i is down ) (j is right)
	if (GetRows() == rhs.GetRows() && rhs.GetCols() == 1) {
		// Edit in row lines
		iStep = 1;
		jNext = 1;
		stepCount = GetRows();
		nextCount = GetCols();
		lineWise = true;
	}

	// (-)
	else if (GetCols() == rhs.GetCols() && rhs.GetRows() == 1) {
		// Edit in col lines
		jStep = 1;
		iNext = 1;
		stepCount = GetCols();
		nextCount = GetRows();
		lineWise = true;
	}

	if (lineWise) {

		// Loop basis swap (i, j) -> next, step
		size_t step = 0;
		size_t next = 0;


		while (next < nextCount) {
			step = 0;
			while (step < stepCount) {

				// switch to i, j basis
				size_t i = iStep * step + iNext * next;
				size_t j = jStep * step + jNext * next;

				// for line, only take into account the (line step) basis, not the (next) basis
				at(i, j) = op(at(i, j), rhs.atC(iStep * step, jStep * step));
				
				++step;
			}

			++next;
		}

		return *this;
	}

	// sentinal
	if (GetCols() != rhs.GetCols() || GetRows() != rhs.GetRows()) {
		std::cout << "Error: Subtraction shapes do not match" << std::endl;
		return *this;
	}

	// edit the data
	for (size_t i = 0; i < GetRows(); ++i) {
		for (size_t j = 0; j < GetCols(); ++j) {
			at(i, j) = op(atC(i, j), rhs.atC(i, j));
		}
	}

	return *this;
}


Tensor Tensor::DotH(const Tensor& o, bool isTransposed, bool isBiased) { 

	// sentinal
	if (GetCols() != o.GetRows()) {
		std::cout << "Error: Dot product dimensions are incompatible" << std::endl;
		return *this;
	}

	// bias
	size_t bias = 0;
	if (isBiased) {
		bias = 1;
	}

	// dot
	Tensor d;
	if (!isTransposed) {
		d.ZeroInit(GetRows() + bias, o.GetCols());
	}
	else {
		d.Transpose();
		d.ZeroInit(GetRows() + bias, o.GetCols());
	}

	for (int i = 0; i < GetRows() + bias; ++i) {
		for (int j = 0; j < o.GetCols(); ++j) {

			// bias (row of 1s)
			if (i >= GetRows()) {
				d.at(i, j) = 1.0f;
				continue;
			}

			float sum = 0;
			for (int k = 0; k < GetCols(); ++k) {
				sum += atC(i, k) * o.atC(k, j);
			}
			d.at(i, j) = sum;
		}
	}

	return d;
}

Tensor& Tensor::operator+=(const Tensor& rhs) { 
	return TwoTensorOp([](float a, float b) { return a + b; }, rhs);
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
	return TwoTensorOp([](float a, float b) { return a - b; }, rhs);
}

Tensor& Tensor::operator*=(const Tensor& rhs) {
	return TwoTensorOp([](float a, float b) { return a * b; }, rhs);
}

Tensor& Tensor::operator/=(const Tensor& rhs) {
	return TwoTensorOp([](float a, float b) { return a / b; }, rhs);
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

Tensor Tensor::Sum(size_t direction) {
	return Reduce(direction, [](float acc, float cur) {
		return cur + acc;
		});
}

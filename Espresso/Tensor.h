#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

class Tensor {

public:

	// static seed
	static void Seed(unsigned int seed) {
		mSeed = seed;
	}

	// constructors
	Tensor() {};
	Tensor(float init);
	// Tensor(const std::vector<std::vector<float>>& init);
	// Tensor(const std::vector<std::vector<float>>& init);
	Tensor(const std::vector<std::vector<float>>& init);
	Tensor(const std::vector<std::vector<int>>& init);

	// destructor
	~Tensor();

	// move constructor
	Tensor(Tensor&& rhs) noexcept;

	// copy constructor
	Tensor(const Tensor& rhs);

	// equality operator
	Tensor& operator=(Tensor& rhs);

	Tensor& operator=(Tensor&& rhs) noexcept;

	// init
	void ZeroInit(size_t r, size_t c);

	// pass in size and a random range
	void RandInit(size_t r, size_t c, float min, float max, unsigned int seed = mSeed);

	// Load in a vector
	void SetData(const std::vector<std::vector<float>>& t);
	std::vector<std::vector<float>> GetData() const;
	

	// Matrix Ops
	inline float& at(size_t i, size_t j) {
		if (i >= GetRows() || j >= GetCols()) {
			std::cerr << "Error: at() passed out of bounds";
			return mData[0][0];
		}
		return (mIsTransposed ? mData[j][i] : mData[i][j]);
	}

	inline float atC(size_t i, size_t j) const {
		if (i >= GetRows() || j >= GetCols()) {
			std::cerr << "Error: at() passed out of bounds";
			return mData[0][0];
		}
		return (mIsTransposed ? mData[j][i] : mData[i][j]);
	}

	// Runs a function on each element of the Tensor
	void ForEach(std::function<void(float&)> f);

	// Provide a funciton that takes in an acc and element and returns the new acc
	Tensor Reduce(size_t axis, std::function<float(float, float)> reducer);

	// Takes a function op (this.element, rhs.element) and another element rhs
	Tensor& LineByLine(std::function<float(float, float)> op, Tensor& rhs);
	Tensor& ElementByElement(std::function<float(float, float)> op, Tensor& rhs);

	// std::vector<float> at(size_t i);
	void Transpose() {
		mIsTransposed = (mIsTransposed != true);
	}

	// math
	Tensor Dot(Tensor o) {
		Tensor t = DotH(o, false);
		return t;
	}

	Tensor DotT(Tensor o) {
		Tensor t = DotH(o, true);
		return t;
	}

	Tensor& operator+=(const Tensor& rhs);
	Tensor& operator-=(const Tensor& rhs);
	Tensor& operator+=(float rhs);
	Tensor& operator-=(float rhs);
	Tensor& operator*=(float rhs);
	Tensor& operator/=(float rhs);

	Tensor Max(size_t direction);
	
	// Getters (by value)
	inline size_t GetRows() const {
		return mIsTransposed ? mCols : mRows;
	}

	inline size_t GetCols() const {
		return mIsTransposed ? mRows : mCols;
	}

	// Utility
	void Print();


private:

	static unsigned int mSeed;

	// Getters (by reference)
	inline size_t& Rows() {
		return mIsTransposed ? mCols : mRows;
	}

	inline size_t& Cols() {
		return mIsTransposed ? mRows : mCols;
	}

	// Helper function for dot product
	Tensor DotH(Tensor o, bool isTransposed);

	// Memory Management
	void AllocTensor(size_t rows, size_t cols);
	void DeallocTensor();

	float** mData = nullptr;
	size_t mRows = 0;
	size_t mCols = 0;

	bool mIsTransposed = false;


};

#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

class Tensor {

public:

	// constructors
	Tensor() {};
	Tensor(float init);
	Tensor(const std::vector<std::vector<float>>& init);
	Tensor(const std::vector<std::vector<int>>& init);

	// destructor
	~Tensor();

	// move constructor
	Tensor(Tensor&& rhs);

	// copy constructor
	Tensor(const Tensor& rhs);

	// equality operator
	Tensor& operator=(Tensor rhs);

	// init
	void ZeroInit(size_t r, size_t c);

	// pass in size and a random range
	void RandInit(size_t r, size_t c, std::function<void()> randGen, size_t seed);

	// Memory management
	void AllocTensor(size_t rows, size_t cols);
	void DeallocTensor();

	// Matrix Ops
	inline float& at(size_t i, size_t j) {
		if (i >= GetRows() || j >= GetCols()) {
			std::cerr << "Error: at() passed out of bounds";
			return mData[0][0];
		}
		return (mIsTranslated ? mData[j][i] : mData[i][j]);
	}

	inline float atC(size_t i, size_t j) const {
		if (i >= GetRows() || j >= GetCols()) {
			std::cerr << "Error: at() passed out of bounds";
			return mData[0][0];
		}
		return (mIsTranslated ? mData[j][i] : mData[i][j]);
	}

	// Runs a function on each element of the Tensor
	void ForEach(std::function<void(float&)> f);

	// Provide a funciton that takes in an acc and element and returns the new acc
	Tensor Reduce(size_t axis, std::function<float(float, float)> reducer);

	// std::vector<float> at(size_t i);
	void Translate() {
		mIsTranslated = (mIsTranslated != true);
	}

	// math
	Tensor Dot(Tensor o);
	Tensor& operator+=(const Tensor& rhs);
	Tensor& operator-=(const Tensor& rhs);
	Tensor& operator+=(float rhs);
	Tensor& operator-=(float rhs);
	Tensor& operator*=(float rhs);
	Tensor& operator/=(float rhs);

	Tensor Max(size_t direction);
	

	// Getters
	inline size_t& Rows() {
		return mIsTranslated ? mCols : mRows;
	}

	inline size_t& Cols() {
		return mIsTranslated ? mRows : mCols;
	}

	inline size_t GetRows() const {
		return mIsTranslated ? mCols : mRows;
	}

	inline size_t GetCols() const {
		return mIsTranslated ? mRows : mCols;
	}

	// Utility
	void Print();


private:

	void SetData(const std::vector<std::vector<float>>& t);

	bool mIsTranslated = false;
	float** mData = nullptr;
	size_t mRows = 0;
	size_t mCols = 0;


};
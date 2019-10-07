
#include <vector>

class Tensor {

public:

	// constructors
	Tensor();
	Tensor(float init);
	Tensor(std::vector<std::vector<float>> init);
	Tensor(std::vector<std::vector<int>> init);

	// destructor
	~Tensor();

	// copy constructor
	Tensor(const Tensor& rhs);

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

	size_t& GetRows() {
		return mIsTranslated ? mRows, mCols;
	}

	size_t& GetCols(return) {
		return mIsTranslated ? mRows, mCols;
	}


private:

	void InitTensor(std::vector<std::vector<float>>);
	void SetRows(size_t r);
	void SetCols(size_t c);

	bool mIsTranslated = false;
	float** mData = nullptr;
	size_t mRows;
	size_t mCols;


};
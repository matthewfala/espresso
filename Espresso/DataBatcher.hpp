#include "Tensor.h"
#include <algorithm>
#include <random>

using std::vector;

struct Data {
	Tensor X;
	Tensor y;
	bool lastBatch = false;
};

class DataBatcher {

public:

	// constructor
	DataBatcher(const vector<vector<int>>& X, const vector<int>& y, size_t batchSize = 1, unsigned int seed = 0) {
		
		mBatchSize = batchSize;

		// convert to float vector
		Tensor tX = Tensor(X);
		tX /= 255; // scale to 0-1;

		vector<vector<int>> yV{ y };
		Tensor tY = Tensor(yV);

		mX = tX.GetData();
		mY = tY.GetData()[0];

		mDataOrder.resize(y.size());
		size_t index = 0;
		for (auto& d : mDataOrder) {
			d = index;
			++index;
		}

		// setup random generator
		mGenerator = new std::mt19937(seed);
		
	}
	
	// Destructor
	~DataBatcher() {
		delete mGenerator;
	}


	// remove copy constructor and assignment op
	DataBatcher& operator=(const DataBatcher&) = delete;
	DataBatcher(const DataBatcher&) = delete;


	// shuffle
	void Shuffle() {
		std::shuffle(mDataOrder.begin(), mDataOrder.end(), *mGenerator);
		mDataNextIndex = 0;
	}

	Data GetBatch() {
		vector<vector<float>> X;
		vector<float> Y;

		if (mDataNextIndex >= mY.size()) {
			std::cerr << "Error: DataBatcher has run out of data. Please reshuffle before reuse." << std::endl;
			return Data{ Tensor(), Tensor() };
		}

		bool lastBatch = false;
		for (size_t i = 0; i < mBatchSize; ++i) {
			size_t index = mDataOrder[mDataNextIndex];
			X.emplace_back(mX[index]);
			Y.emplace_back(mY[index]);
			++mDataNextIndex;

			// All data consumed
			if (mDataNextIndex >= mY.size()) {
				lastBatch = true;
				break;
			}
		}

		Tensor tX(X);
		tX.Transpose();
		Tensor tY(vector<vector<float>>{ Y });

		// maybe transpose
		// tY.Transpose();
		return Data{ tX, tY, lastBatch };

	}

	size_t GetDataCount() const {
		return mY.size();
	}

private:

	vector<vector<float>> mX;
	vector<float> mY;
	size_t mBatchSize;
	vector<size_t> mDataOrder;
	size_t mDataNextIndex = 0;
	std::mt19937* mGenerator;
};
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

		// convert to float vector
		Tensor tX = Tensor(X);

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
		for (int i = 0; i < mBatchSize; ++i) {
			X.emplace_back(mX[i]);
			Y.emplace_back(mY[i]);
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
		tY.Transpose();
		return Data{ tX, tY, lastBatch };

	}

private:

	vector<vector<float>> mX;
	vector<float> mY;
	size_t mBatchSize;
	vector<size_t> mDataOrder;
	size_t mDataNextIndex = 0;
	std::mt19937* mGenerator;
};
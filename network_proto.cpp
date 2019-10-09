
#include <vector>
#include <iostream>
#include <limits>

#include "MNIST_reader.h"
#include "Espresso/Tensor.h"
#include "Espresso/DataBatcher.hpp"

using std::vector;

// Function prototypes
void LoadData(vector<vector<int>>& training_images, vector<int>& training_labels);
void AddBiasBit(vector<vector<int>>& training_images);

// Sigmoid
auto g = [](float& x) { return 1.0f / (1.0f + exp(-x)); };
auto gprime = [](float& y) { return y * (1 - y); };

int main()
{
	// Config
	size_t batchSize = 4;
	size_t layers = 2;
	vector<size_t> hiddenLayerSizes{ 100, 10 }; // +1 for weight due to bias

	// Load data
	vector<vector<int>> training_images;
	vector<int> training_labels;
	LoadData(training_images, training_labels);
	AddBiasBit(training_images);  // bias trick

	size_t inputLayerSize = training_images[0].size() - 1;
	vector<size_t> layerSizes(hiddenLayerSizes);
	layerSizes.emplace(layerSizes.begin(), inputLayerSize);

	// Initiallize network
	vector<Tensor> weights;
	vector<Tensor> weightGrads;
	vector<Tensor> layerOutputCache;
	for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
		weights.emplace_back(Tensor());
		weightGrads.emplace_back(Tensor());

		// +1 for bias trick
		weights[i].RandInit(layerSizes[i + 1], layerSizes[i] + 1, -.05, .05);
		weightGrads[i].ZeroInit(layerSizes[i + 1], layerSizes[i] + 1);
	}

	// Load data batcher
	DataBatcher batcher(training_images, training_labels, batchSize);
	batcher.Shuffle();

	// Get batch (& unpack)
	Data batch = batcher.GetBatch();
	Tensor X = batch.X;
	Tensor y = batch.y;
	bool isLast = batch.lastBatch;

	// Forward pass

	// FC & Sigmoid
	Tensor O = X;
	for (size_t layer = 0; layer < weights.size(); ++layer) {

		// FC
		std::cerr << "FC Layer: " << layer << std::endl;
		Tensor& W = weights[layer];
		W.PrintShape();
		O.PrintShape();

		if (layer != weights.size() - 1) {
			O = W.DotTB(O); // use DotTB for a pretransposed matrix + bias bit
		}
		else {
			O = W.DotT(O); // last layer's output is not biased
		}

		O.PrintShape();

		// Sigmoid
		O.ForEach(g);

		// save the output
		layerOutputCache.emplace_back(O);
	}


	std::cerr << "Running Deviation" << std::endl;
	// Deviation
	Tensor squareInput = layerOutputCache.back();
	squareInput.Print();

	squareInput *= -1;
	squareInput.Print();
	y.Print();
	squareInput += y; // row wise addition

	squareInput.Print();
	y.PrintShape();

	// Square & loss
	Tensor square = squareInput;  // square by element

	std::cerr << "SQUARE " << std::endl;
	square.Print();
	square *= square;

	square.Print();
	Tensor loss = square.Sum(0).Sum(1);


	// Log result
	std::cerr << "Loss: " << loss.at(0, 0) << std::endl;





	/*
	inline double g(double x) { return 1.0 / (1.0 + exp(-x)); }
	inline double gprime(double y) { return y * (1 - y); }
	
	*/


	


	return 0;
}





void LoadData(vector<vector<int>>& training_images, vector<int>& training_labels) {

	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	loadMnistLabels(filename, training_labels);
	cout << "Number of labels: " << training_labels.size() << endl;
	
}

void AddBiasBit(vector<vector<int>>& training_images) {
	for (auto t : training_images) {
		t.emplace_back(1.0f);
	}
}







// Skeleton Code
/*
// Data contains batchsize
Records Train(Network& n, int epochs=100, Optimizer& optimizer=SGD, const DataBatcher& dataBatcher, const Data& valData) {

	Records records;
	float minValLoss = std::numeric_limits<T>::max;

	// train loop
	for (int e = 0; e < epochs, ++e) {

		Data batch;
		while (DataBatcher >> batch) {

			// Forward & Backward pass
			n.ForwardPass(batch); // discard output tensor
			n.BackwardPass();

			// Update
			n.Update(optimizer);

		}

		// Test the data
		Data trainData = dataBatcher.All();
		Record trainRec = Test(n, trainData);
		Record valRec = Test(n, valData);

		// save record
		records.Append(trainRec, valRec);
		if (valRec.loss < minValLoss) {
			Records.SaveNetwork(n);
		}

	}

	return records;
}



Record Test(Network n, Data data) {

	// Forward pass


	return;
}

void Predict(Network n, Unlabeled ud) {
	return;
}

*/

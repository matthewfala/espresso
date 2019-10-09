
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
	// 	O.PrintShape();

	for (size_t layer = 0; layer < weights.size(); ++layer) {

		// FC
		std::cerr << "FC Layer: " << layer << std::endl;
		Tensor& W = weights[layer];
		if (layer != weights.size() - 1) {
			O = W.DotTB(O); // use DotTB for a pretransposed matrix + bias bit
			O.ToggleBiasOff(); // temporarily remove bias for sigmoid
		}
		else {
			O = W.DotT(O); // last layer's output is not biased
		}

		// Sigmoid
		O.ForEach(g);
		O.ToggleBiasOn(); // add back bias (if present);

		O.Print();

		// save the output
		layerOutputCache.emplace_back(O);

		// output progress
		// O.PrintShape();
	}

	// Deviation
	Tensor squareInput = layerOutputCache.back();
	squareInput *= -1;
	squareInput += y; // row wise addition

	// Square & loss
	Tensor square = squareInput;  // square by element
	square *= square;
	Tensor loss = square.Sum(0).Sum(1);
	loss /= static_cast<float>(batchSize);


	// Log result
	std::cerr << "Loss: " << loss.at(0, 0) << std::endl;


	// Backprop

	// initiallize the output gradient square
	Tensor squareOutputGrad;
	squareOutputGrad.ZeroInit(square.GetRows(), square.GetCols());
	square.Print();
	square += 10000;
	square.Print();

	// reverse sum
	squareOutputGrad += (1.0f / batchSize);
	Tensor squareLocalGrad = squareInput;
	squareLocalGrad *= 2;

	// loss function
	// dInput = dOutput * LocalGradient
	Tensor lossInputGrad = squareOutputGrad;
	lossInputGrad *= squareLocalGrad;
	lossInputGrad *= -1;

	lossInputGrad.PrintShape();
	lossInputGrad.Print();

	// layer backprop
	Tensor layerOutputGradient = lossInputGrad;
	for (int oi = layerOutputCache.size() - 1; oi >= 0; --oi) {

		// Reverse sigmoid

		// local gradient
		Tensor& O = layerOutputCache[oi];
		if (oi != layerOutputCache.size() - 1) {
			O.ToggleBiasOff(); // remove the bias bit if not output layer
		}

		Tensor dO = O;
		dO.ForEach(gprime);
		std::cerr << "Rows: " << dO.GetRows() << std::endl;

		std::cerr << "Next Layer Input Gradient" << std::endl;
		layerOutputGradient.PrintShape();
		std::cerr << "Local Sigmoid Gradient" << std::endl;
		dO.PrintShape();

		// global gradient
		dO *= layerOutputGradient;

		// Reverse FC
		Tensor& Xi = (oi > 0) ? layerOutputCache[oi - 1] : X;
		
		Xi.Transpose();
		Tensor dW = O.Dot(Xi);
		Xi.Transpose();

		Tensor& W = weights[oi];
		W.Transpose();
		layerOutputGradient = W.Dot(O);
		layerOutputGradient.ToggleBiasOff();
		W.Transpose();

		// restore O
		O.ToggleBiasOn();
		
		layerOutputGradient.PrintShape();
		
	}


	


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

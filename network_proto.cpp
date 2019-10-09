
#include <vector>
#include <iostream>
#include <limits>

#include "MNIST_reader.h"
#include "Espresso/Tensor.h"
#include "Espresso/DataBatcher.hpp"

using std::vector;

// Struct declarations
struct Network {
	vector<Tensor> weights;
	vector<Tensor> weightGrads;
	vector<Tensor> layerOutputCache;

	// backprop cache
	Tensor squareInput, square;
	
	// data cache
	Tensor loss;
	float totalLoss;

	// prediction store
	Tensor predictions;
	float correctPredictions;
	float totalPredictions;
};

struct Record {
	float loss = 0.0f;
	int correct = 0;
	int totalPredictions = 0;
};

// Function prototypes
void LoadData(vector<vector<int>>& training_images, vector<int>& training_labels);
void AddBiasBit(vector<vector<int>>& training_images);
void BatchTrain(Network& n, Data batch, float lr);
Tensor OneHotEncode(Tensor& y);
void ForwardPass(Network& n, const Tensor& X, const Tensor& y, bool predictionMode = false);
Record Test(Network& n, DataBatcher& batcher);

// Sigmoid
auto g = [](float& x) { x = 1.0f / (1.0f + exp(-x)); };
auto gprime = [](float& y) { y = y * (1 - y); };

int main()
{

	// Data partitions
	int trainSize = 6000;
	int testSize = 10000;

	float testToVal = 1.0f / 3;

	// Config
	float lr = .1;
	int epochs = 500;
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


	// Partition data

	// Training set
	vector<vector<int>> training_set_img(training_images.begin(), training_images.begin() + trainSize);
	vector<int> training_set_lbl(training_labels.begin(), training_labels.begin() + trainSize);

	// Testing set
	vector<vector<int>> testing_set_img(training_images.begin() + trainSize, training_images.begin() + trainSize + testSize);
	vector<int> testing_set_lbl(training_labels.begin() + trainSize, training_labels.begin() + trainSize + testSize);
	DataBatcher testBatcher(testing_set_img, testing_set_lbl, batchSize);

	// Training: Train partition
	int trainSetSize = trainSize * testToVal;
	vector<vector<int>> train_img(training_set_img.begin(), training_set_img.begin() + trainSetSize);
	vector<int> train_lbl(training_set_lbl.begin(), training_set_lbl.begin() + trainSetSize);
	DataBatcher trainBatcher(testing_set_img, testing_set_lbl, batchSize);

	// Training: Val partition
	vector<vector<int>> val_img(training_set_img.begin() + trainSetSize, training_set_img.end());
	vector<int> val_lbl(training_set_lbl.begin() + trainSetSize, training_set_lbl.end());
	DataBatcher valBatcher(testing_set_img, testing_set_lbl, batchSize);


	// Initiallize network
	Network nn;
	for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
		nn.weights.emplace_back(Tensor());
		nn.weightGrads.emplace_back(Tensor());

		// +1 for bias trick
		nn.weights[i].RandInit(layerSizes[i + 1], layerSizes[i] + 1, -.05, .05);
		// weightGrads[i].ZeroInit(layerSizes[i + 1], layerSizes[i] + 1); // Init later
	}

	// Load data batcher
	// DataBatcher batcher(training_images, training_labels, batchSize);

	// Records
	std::vector<Record> trainRecords;
	std::vector<Record> valRecords;
	Record testRecord;

	for (int e = 0; e < epochs; ++e) {
		// shuffle data
		trainBatcher.Shuffle();

		// train loop
		bool lastBatch = false;
		do {

			// Get batch (& unpack)
			Data batch = trainBatcher.GetBatch();
			lastBatch = batch.lastBatch;
			BatchTrain(nn, batch, lr);

		} while (!lastBatch);

		// Test & record
		trainRecords.emplace_back(Test(nn, trainBatcher));
		valRecords.emplace_back(Test(nn, valBatcher));
		
	}

	testRecord = Test(nn, testBatcher);

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


// expects a tensor of labels with one row
Tensor OneHotEncode(Tensor& y, size_t catigories) {
	if (y.GetRows() != 1) {
		std::cerr << "ERROR: Please provide OneHotEncode with a tensor of 1 row" << std::endl;
		return Tensor();
	}

	Tensor hot;
	//size_t max = static_cast<size_t>(y.Max(1).atC(0, 0));
	hot.ZeroInit(catigories, y.GetCols());

	for (int col = 0; col < y.GetCols(); ++col) {
		int colOneInd = static_cast<int>(y.at(0, col));
		hot.at(colOneInd, col) = 1;
	}

	return hot;
}

void AddBiasBit(vector<vector<int>>& training_images) {
	for (auto t : training_images) {
		t.emplace_back(1.0f);
	}
}


Record Test(Network& n, DataBatcher& batcher) {
	// shuffle the data
	batcher.Shuffle();

	// Test loop (all data)
	Record r;

	bool lastBatch = false;
	do {

		// Get batch (& unpack)
		Data batch = batcher.GetBatch();
		lastBatch = batch.lastBatch;
		
		Tensor X = batch.X;
		Tensor y = batch.y;
		y = OneHotEncode(y, 10); // convert y to One hot encoding

		ForwardPass(n, X, y, true);
		r.correct += n.correctPredictions;
		r.totalPredictions += n.totalPredictions;
		r.loss += n.totalLoss;
		
	} while (!lastBatch);

	return r;
}

void BatchTrain(Network& n, Data batch, float lr) {

	Tensor X = batch.X;
	Tensor y = batch.y;
	y = OneHotEncode(y, 10); // convert y to One hot encoding

	bool isLast = batch.lastBatch;

	// Forward pass
	ForwardPass(n, X, y);

	// Log result
	std::cerr << "Loss: " << n.loss.at(0, 0) << std::endl;

	// Backprop

	// loss function

	// initiallize the output gradient square
	Tensor squareOutputGrad;
	squareOutputGrad.ZeroInit(n.square.GetRows(), n.square.GetCols());
	
	// reverse sum
	squareOutputGrad += (1.0f / X.GetCols());   // Possible error source

	Tensor squareLocalGrad = n.squareInput;
	squareLocalGrad *= 2;

	// Good

	// dInput = dOutput * LocalGradient
	Tensor lossInputGrad = squareOutputGrad;
	lossInputGrad *= squareLocalGrad;
	lossInputGrad *= -1;
	// lossInputGrad.PrintShape();

	// std::cerr << "Losss input grad" << std::endl;
	// lossInputGrad.Print();

	// layer backprop
	Tensor layerOutputGradient = lossInputGrad;
	for (int oi = static_cast<int>(n.layerOutputCache.size() - 1); oi >= 0; --oi) {

		// Reverse sigmoid

		// local gradient
		Tensor& O = n.layerOutputCache[oi];
		if (oi != n.layerOutputCache.size() - 1) {
			O.ToggleBiasOff(); // remove the bias bit if not output layer
		}

		Tensor dO = layerOutputGradient;
		// dO.Print();

		dO.ForEach(gprime);
		// dO.Print();

		// std::cerr << "Rows: " << dO.GetRows() << std::endl;

		// std::cerr << "Next Layer Input Gradient" << std::endl;
		// layerOutputGradient.PrintShape();
		// std::cerr << "Local Sigmoid Gradient" << std::endl;
		// dO.PrintShape();                                          // Checked to here.

		// global gradient
		dO *= layerOutputGradient;

		// Reverse FC
		Tensor& Xi = (oi > 0) ? n.layerOutputCache[static_cast<size_t>(oi) - 1] : X;

		Xi.Transpose();
		n.weightGrads[oi] = O.Dot(Xi);

		Xi.Transpose();

		Tensor& W = n.weights[oi];
		W.Transpose();
		layerOutputGradient = W.Dot(O);
		layerOutputGradient.ToggleBiasOff();
		W.Transpose();

		// restore O
		O.ToggleBiasOn();

		//layerOutputGradient.PrintShape();

	}


	// Gradient Descent (Update W)
	for (int i = 0; i < n.weights.size(); ++i) {

		// Grad -> Step size
		n.weightGrads[i] *= lr;
		n.weights[i] -= n.weightGrads[i];

	}

}


// FC & Sigmoid
void ForwardPass(Network& n, const Tensor& X, const Tensor& y, bool predictionMode) {

	Tensor O = X;

	// remove output cache
	n.layerOutputCache.clear();

	for (size_t layer = 0; layer < n.weights.size(); ++layer) {

		// FC
		std::cerr << "FC Layer: " << layer << std::endl;
		Tensor& W = n.weights[layer];
		if (layer != n.weights.size() - 1) {
			O = W.DotTB(O); // use DotTB for a pretransposed matrix + bias bit
			O.ToggleBiasOff(); // temporarily remove bias for sigmoid
		}
		else {
			O = W.DotT(O); // last layer's output is not biased
		}

		// Sigmoid
		O.ForEach(g);
		O.ToggleBiasOn(); // add back bias (if present);

		// save the output
		n.layerOutputCache.emplace_back(O);

		// output progress
		// O.PrintShape();
	}

	// Predict
	if (predictionMode) {
		Tensor labelMax = n.layerOutputCache.back().Max(0);
		Tensor labelOneHot = labelMax.TwoTensorOp([](float a, float b) {
			if (a == b) {
				return 1.0f;
			}
			return 0.0f;
			},
			labelMax);

		n.predictions = labelOneHot;

		Tensor accuracy = labelOneHot.TwoTensorOp(
			[](float a, float b) {
				if (a == b) {
					return 1.0f;
				}
				return 0.0f;
			},
			y);
		accuracy = accuracy.Sum(0).Sum(1);

		n.correctPredictions = accuracy.atC(0, 0);
		n.totalPredictions = y.GetCols();
	}
	

	// Loss Function
	n.squareInput = n.layerOutputCache.back();
	n.squareInput *= -1;
	n.squareInput += y; // element wise addition

	// Square & loss
	n.square = n.squareInput;  // square by element
	n.square *= n.square;
	n.loss = n.square.Sum(0).Sum(1);
	n.totalLoss = n.loss.atC(0, 0); // for records
	n.loss /= static_cast<float>(X.GetCols());   //// POSSIBLE ERROR SOURCE!!!!!!!
	return;
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

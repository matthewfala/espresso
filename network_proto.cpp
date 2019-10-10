
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

	float testToVal = 2.0f / 3;

	// Config
	float lr = 1;
	int epochs = 35;
	size_t batchSize = 32;
	size_t layers = 3;
	vector<size_t> hiddenLayerSizes{ 32, 32, 10 }; // +1 for weight due to bias

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
	DataBatcher trainBatcher(train_img, train_lbl, batchSize);

	// Training: Val partition
	vector<vector<int>> val_img(training_set_img.begin() + trainSetSize, training_set_img.end());
	vector<int> val_lbl(training_set_lbl.begin() + trainSetSize, training_set_lbl.end());
	DataBatcher valBatcher(val_img, val_lbl, batchSize);


	std::cerr << std::endl;
	std::cerr << "Partitioned Data" << std::endl;
	std::cerr << "Test Set Size: 	" << testing_set_img.size() << std::endl;
	std::cerr << "Train Set Size:	" << train_img.size() << std::endl;
	std::cerr << "Val Set Size:		" << val_img.size() << std::endl;
	std::cerr << std::endl;


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

		std::cerr << std::endl << std::endl << "Epoch: " << e << std::endl;
		// shuffle data
		trainBatcher.Shuffle();

		// train loop
		bool lastBatch = false;
		do {

			// Get batch (& unpack)
			Data batch = trainBatcher.GetBatch();
			lastBatch = batch.lastBatch;
			BatchTrain(nn, batch, lr);

			std::cerr << ("X");

		} while (!lastBatch);

		std::cerr << std::endl;

		// Test & record
		std::cerr << "Recording Train Performance" << std::endl;
		trainRecords.emplace_back(Test(nn, trainBatcher));
		std::cerr << "Loss: " << trainRecords.back().loss << std::endl;
		std::cerr << "Accuracy: " << static_cast<float>(trainRecords.back().correct) / trainRecords.back().totalPredictions << std::endl;

		std::cerr << "Recording Val Performance" << std::endl;
		valRecords.emplace_back(Test(nn, valBatcher));
		std::cerr << "Loss: " << valRecords.back().loss << std::endl;
		std::cerr << "Accuracy: " << static_cast<float>(valRecords.back().correct) / valRecords.back().totalPredictions << std::endl;

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

	// Backprop

	// loss function

	// initiallize the output gradient square
	Tensor squareOutputGrad;
	squareOutputGrad.ZeroInit(n.square.GetRows(), n.square.GetCols());
	
	// reverse sum
	squareOutputGrad += (1.0f / X.GetCols());   // Possible error source

	squareOutputGrad.PrintOne("Point A");

	Tensor squareLocalGrad = n.squareInput;
	squareLocalGrad *= 2;

	n.squareInput.PrintOne("x^2 X:");

	squareLocalGrad.PrintOne("Point B*: Chk");




	// Good

	// dInput = dOutput * LocalGradient
	Tensor lossInputGrad = squareOutputGrad;
	lossInputGrad *= squareLocalGrad;
	lossInputGrad *= -1;  // squareInput is negated

	lossInputGrad.PrintOne("Point C:");

	// lossInputGrad.PrintShape();

	// std::cerr << "Losss input grad" << std::endl;
	// lossInputGrad.Print();

	// layer backprop
	Tensor layerOutputGradient = lossInputGrad;
	for (int oi = static_cast<int>(n.layerOutputCache.size() - 1); oi >= 0; --oi) {

		// Reverse sigmoid

		// local gradient
		Tensor& O = n.layerOutputCache[oi]; // sigmoid output
		if (oi != n.layerOutputCache.size() - 1) {
			O.ToggleBiasOff(); // remove the bias bit if not output layer
		}
		
		// start with the sigmoid output
		Tensor dO = O;  //layerOutputGradient;
		dO.PrintOne("Point S:");

		// dO.Print();

		dO.ForEach(gprime); // SHould be called on the output of sigmoid

		dO.PrintOne("Point Delta Sigma");

		// global gradient
		dO *= layerOutputGradient;

		dO.PrintOne("Point d0");

		// Checked up to here.

		// Reverse FC
		Tensor& Xi = (oi > 0) ? n.layerOutputCache[static_cast<size_t>(oi) - 1] : X;

		Xi.Transpose();
		n.weightGrads[oi] = dO.Dot(Xi);

		Xi.Transpose();

		Tensor& W = n.weights[oi];
		W.Transpose();
		layerOutputGradient = W.Dot(dO); // From Christ
		layerOutputGradient.ToggleBiasOff();
		W.Transpose();

		// restore O
		dO.ToggleBiasOn();

		// layerOutputGradient.PrintShape();

	}


	// Gradient Descent (Update W)
	for (int i = 0; i < n.weights.size(); ++i) {
		//n.weightGrads[i].Print();
		//w.Print();

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
		// std::cerr << "FC Layer: " << layer << std::endl;
		Tensor& W = n.weights[layer];
		if (layer != n.weights.size() - 1) {
			O = W.DotTB(O); // use DotTB for a pretransposed matrix + bias bit
			O.ToggleBiasOff(); // temporarily remove bias for sigmoid
		}
		else {
			O = W.DotT(O); // last layer's output is not biased
		}

		// O.Print();
		//std::cerr << "waiting" << std::endl;


		// Sigmoid
		O.ForEach(g);
		O.ToggleBiasOn(); // add back bias (if present);

		O.PrintOne("Sigmoid Output:");


		// save the output
		if (!predictionMode) {
			n.layerOutputCache.emplace_back(O);
		}

		// output progress
		// O.PrintShape();
	}

	// Predict
	if (predictionMode) {
		/*
		Tensor labelMax = O.Max(0);
		Tensor labelMax
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

			*/

		Tensor accuracy = O.MaxIndiceByRow().TwoTensorOp([](float a, float b) {
			if (a == b) {
				return 1;
			};
			return 0;
			}, y.MaxIndiceByRow());
		

		n.correctPredictions = accuracy.Sum(1).atC(0, 0);
		n.totalPredictions = y.GetCols();
	}

	if (!predictionMode) {

		// Loss Function
		n.squareInput = n.layerOutputCache.back();
		n.squareInput *= -1;
		n.squareInput += y; // element wise addition
		//y.Print();

		// Square & loss
		n.square = n.squareInput;  // square by element
		n.square *= n.square;
		n.loss = n.square.Sum(0).Sum(1);
		n.totalLoss = n.loss.atC(0, 0); // for records
		n.loss /= static_cast<float>(X.GetCols()); 

	}
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


#include <vector>
#include <iostream>
#include <limits>

#include "MNIST_reader.h"
#include "Espresso/Tensor.h"
#include "Espresso/DataBatcher.hpp"

using namespace std;
int main()
{
	
	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int> > training_images;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cerr << "Image size: " << training_images[0].size() << endl;

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	vector<int> training_labels;
	loadMnistLabels(filename, training_labels);
	cerr << "Number of labels: " << training_labels.size() << endl;

	
	// Initiallize our network

	vector<vector<int>> training_images_test{ {1, 2, 3}, { 4, 5, 6 }};
	vector<int> training_labels_test{ 1, 0 };


	DataBatcher db(training_images, training_labels, 4);
	db.Shuffle();

	Data batch1 = db.GetBatch();

	std::cerr << "Data y" << std::endl;
	batch1.y.Print();
	
	std::cerr << "Data X" << std::endl;
	batch1.X.Print();

	std::cerr << "Data Done" << std::endl;
	std::cerr << batch1.lastBatch << std::endl;


	std::cerr << "Batch 2" << std::endl;
	Data batch2 = db.GetBatch();

	std::cerr << "Data y" << std::endl;
	batch2.y.Print();

	std::cerr << "Data X" << std::endl;
	batch2.X.Print();

	std::cerr << "Data Done" << std::endl;
	std::cerr << batch2.lastBatch << std::endl;


	

	// create nodes
	


	return 0;
}

// Skeleton Code
/*
// Data contains batchsize
Records Train(Network& n, int epochs = 100, Optimizer& optimizer = SGD, const DataBatcher& dataBatcher, const Data& valData) {

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

*/
// Network

// Automatic indexing
// On
// AddNode(NetNode) <- Called from NetNode constructor, adds to NetNode* vector,  calles SetNode on NetNode (indice of vector)
//

// SetLoss
// SetOutput
// ForwardPass
// BackwardsPass
// NetNode LossNode (chain loss function to this)
// NetNode OutputNode (chain d

/*
Record Test(Network n, Data data) {

	// Forward pass


	return;
}

void Predict(Network n, Unlabeled ud) {
	return;
}

*/

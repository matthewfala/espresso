#include <vector>
#include <iostream>
#include <limits>

#include "MNIST_reader.h"

using namespace std;
int main()
{
	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< int> > training_images;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	vector<int> training_labels;
	loadMnistLabels(filename, training_labels);
	cout << "Number of labels: " << training_labels.size() << endl;


	// Initiallize our network

	// create nodes
	

	std::vector<NetNode> layers {
		NN
	}


	return 0;
}

// Skeleton Code

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


Record Test(Network n, Data data) {

	// Forward pass


	return;
}

void Predict(Network n, Unlabeled ud) {

}
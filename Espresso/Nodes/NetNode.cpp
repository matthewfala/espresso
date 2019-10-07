
#include "NetNode.h"



// Recieve input from forward pass
void NetNode::ForwardRecieve(Tensor t, const NetNode& supplier) {

	// add tensor to store
	bool inputFull = true;
	for (auto ii : InputStore) {
		
		// update cache
		if (ii.supplierID == supplier.id) {
			ii.data = t;
			ii.isLoaded = true;
		}
		else if (ii.isLoaded == false) {
			inputFull = false;
		}
	}

	// if all inputs recieved, process
	if (inputFull) {

		// Process
		
		// Pass
	}

}

void NetNode::BackwardsRecieve(const NetNode& n, InputType) {

}

void NetNode::Chain(const NetNode& n, InputType inputType) {


	// Add to output list
	mOutputNodes.emplace_back(n.id);

	// Add node to other nodes input list
	n.AddInput(id, inputType);
	return;

}


void NetNode::AddInput(size_t id, InputType inputType) {
	InputInfo ii{
			inputType,
			id,
			Tensor(); // holder tensor
	}

	inputStore.emplace_back(ii);
}

NetNode* NetNode::GetNode(size_t id) {
	return 

}
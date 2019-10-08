
class Network {
	
public:

	Network() {

	}

	~Network() {
		// Clean up nodes. Delete each in the vector;
		for (auto n : node) {
			delete n;
		}
	}

	// AddNode(NetNode) <- Called from NetNode constructor, adds to NetNode* vector,  calles SetNode on NetNode (indice of vector)
	void AddNode(NetNode* n);

	// Setup
	void SetX(NetNode* n); 
	void SetY(NetNode* n);

	// Forward pass starts with sourceX and sourceY, call forwardRecieve (or a new function to send out the data), and pass a tensor.
	Tensor ForwardPass(Data batch); //  Returns the Sink
	Tensor BackwardsPass(); // Returns Gradient

	// Use optimizer to update
	void Update(Optimizer* opt);


	// Read output
	float GetLoss();
	float GetPrediction();
	float GetInputGradient();

private:
	std::vector<NetNode*> nodes;

	// these nodes are determined automatically
	NetSource* mSourceX = nullptr;  // Data
	NetSource* mSourceY = nullptr;  // Labels

	NetSink* mSink = nullptr; // Starts backrprop

	// Test
	NetReader* mLabels = nullptr; // Reads the output labels before loss function

	// We need some node to read the labels;



};


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
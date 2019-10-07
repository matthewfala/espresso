
class Network {
	
public:
	// AddNode(NetNode) <- Called from NetNode constructor, adds to NetNode* vector,  calles SetNode on NetNode (indice of vector)
	void AddNode(NetNode* n);

	// Setup
	void SetX(NetNode* n); 
	void SetY(NetNode* n);

	// Forward pass starts with sourceX and sourceY, call forwardRecieve (or a new function to send out the data), and pass a tensor.
	Tensor ForwardPass(); //  Returns the Sink
	void BackwardsPass(); // 

	// Use optimizer to update
	void Update(Optimizer* opt);


	// Read output
	float GetLoss();
	float GetPrediction();
	float GetInputGradient();



private:
	std::vector<NetNode*> nodes;

	// these nodes are determined automatically
	NetSource* sourceX = nullptr;  // Data
	NetSource* sourceY = nullptr;  // Labels

	NetSink* sink = nullptr; // Starts backrprop

	// Test
	NetReader* labels = nullptr; // Reads the output labels before loss function

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
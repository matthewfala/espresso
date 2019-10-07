#include <vector>
#include <string>

// network automaically chains the input output nodes based on these flags
enum NodeFlag {
	Standard,
	ReceptorX,
	ReceptorY,
	OutputY,
};

enum InputType {
	X,
	Y,

};

struct InputInfo {
	InputType type;
	size_t supplierID;
	bool isLoaded;
	Tensor data;
};

class NetNode {

public:

	size_t id; // must initiallize in the constructor

	// configure the node in the network
	NetNode(Network& n, std::vector<NodeFlag> nf) {

		flags = nf;
		n.AddNode(this);

	}

	void Chain(NetNode n, InputType inputType);

	// Forward pass
	void ForwardRecieve(Tensor t, const NetNode& supplier);
	void ForwardSend();

	// Process
	virtual void ForwardProcess();
	virtual void BackwardProcess();

	// Send Tensors (Create placeholder for gradient)

	// Back propagation
	void BackwardsRecieve(Tensor grad);
	void BackwardsSend();

	// Update based on optimizer
	virtual void Update();

	std::vector<NodeFlag> GetFlags();
	
	

private:

	// Retain what the significance of the vector is
	std::vector<NodeFlag> flags;

	Network* network
	std::vector<InputInfo> inputStore;
	Tensor gradientSummary;
	bool isGradientInit = false;
	std::vector<size_t> mOutputNodes;
	
	static NetNode* GetNode(size_t id);

	// not to be called by user
	AddInput(const NetNode& n, InputType inputType);




};

class NetNode {

	NetNode() {

	}

	~NetNode() {

	}

	virtual Tensor ForwardPass(Data data) = 0;
	virtual void BackwardsPass();

};
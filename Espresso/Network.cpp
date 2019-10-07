#include "Network.h"

Tensor& ForwardPass(Data batch) {

	// Data consists of at least two Tensors, 
	// 1 X -- Input
	// 2 y -- Labels

	// Pass in the data
	mSourceX.ForwardPass(Data.X);
	mSourceY.ForwardPass(Data.y);

	return mSink.GetOutput();

}

Tensor& BackwardsPass() {
	
	// constructor
	mSink.BackwardPass(Tensor(1.0f));
	return mSourceX.GetGradient();


}
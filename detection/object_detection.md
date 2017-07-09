## 2017-7-9
### R-FCN: Object Detection via Region-based Fully Convolutional Networks
###### published: NIPS 2016
###### author: MSAR Jifeng Dai
> However as empirically investigated in this work, this naive solution turns out to have considerably *inferior detection accuracy* that does not match the network's *superior classification accuracy*.

> To remedy this issue, in the ResNet paper the ROI pooling layer of the Faster R-CNN detector is *unnaturally* inserted between two sets of convolutional layers --- this creates a deeper RoI-wise subnetwork that improves accuracy, at the cost of lower speed due to the unshared per-RoI computation.

**We argue that the aforementioned unnatural design is caused by a dilemma of increasing translation *invariance* for image classification vs. respecting translation *variance* for object detection.**

The key idea of R-FCN for object detection as this fingure:


> We hypothesize that deeper convolutional layers in an image classification network are less sensitive to translation. To address this dilemma, the ResNet paper's detection pipeline inserts the Roi pooling layer into convolutions---this *region-specific* operation breaks down translation invariance, and the post-RoI convolutional layers are no longer translation-invariant when evaluated across different regions. However, this design sacrifices training and testing efficiency since it introduces a considerably number of region-wise layers.

The main architecture of R-FCN is:
1. A RPN layer which is the same as Faster-RCNN. This layer proposal many rois, which is the potential object location.
2. A R-FCN layer which contains $k^2*(C+1)$ feature maps. This layer and RPN layer share the same convolution layer.
3. **Position-sensitive score maps & Position-sensitive RoI pooling**
> We divide each RoI rectangle into $k*k$ bins by a regulay grid. For an RoI rectangle of a size $w*h$, a bin is of a size $ \frac{w}{k}* \frac{h}{k} $. The last convolutional layer is constructed to produce $k^2$ score maps for each category. Inside the (i, j)-th bin $0<=i,j<=k-1$, we define a position-sensitive RoI pooling operation that pools only over the (i,j)-th score map:

$$
r_c(i,j|\Theta) = \sum_{(x,y)\subset{bin(i,j)}}z_{i,j,c}(x+x_0, y+y_0|\Theta)/n
$$

> (x0, y0) denotes the top-left corner of an RoI.

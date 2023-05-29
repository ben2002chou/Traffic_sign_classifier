# Traffic_sign_classifier
A classifier model implemented with Keras trained to classify images. Goal is to have validation accuracy above 93%. 
### As of most recent commit, I have achieved validation accuracy of 96.5% with ResNet18 (test accuracy (94.5%))

I plan to use this notebook to test new CNN architectures I come across in my academia journey (sort of like a fun showdown of models)
Training data is too large to upload. You may have to use your own. Valid/Test data is provided to verify results

### This code uses MacOS GPU enabled tensorflow with Metal backend
 



### Traffic Sign Classifier




I. Introduction
A. Background on the problem of image classification 
Image classification is the task of categorizing images into predefined classes. Early methods involved manual feature extraction. However, these methods were often costly to compute and tend to not generalize well. To expand this task to more complex datasets, Convolutional Neural Networks (CNNs) were used, allowing for automatic feature extraction and better generalization towards images. 	
B. Objective of the paper
The objective of this paper is to implement CNN architectures that were the breakthrough of their time to demonstrate firsthand the benefits and downside to different architectures. We also implement transfer learning on this dataset to see when it is beneficial to do so.
II. Methodology
	Description our implementation of LeNet:
	Input: The input to LeNet is a 32x32 grayscale image.
	C1 - Convolutional layer: This layer has 6 feature maps, each of which is connected to a 5x5 neighborhood in the input, resulting in 28x28 dimensions.
	S2 - Average Pooling layer (subsampling): This layer subsamples each feature map to a 14x14 dimension.
	C3 - Convolutional layer: This layer has 16 feature maps and each unit is connected to several 5x5 neighborhoods at identical locations in a subset of S2's feature maps, resulting in 10x10 dimensions.
	S4 - Average Pooling layer (subsampling): This layer subsamples each C3 feature map to 5x5 dimensions.
	C5 - Convolutional layer: This layer is fully connected to S4, with 120 feature maps each of size 1x1.
	F6 - Fully connected layer: This layer has 84 units.
	Output - The output layer is a fully connected layer with 43 units, corresponding to the 43 sign types.
The initial architecture we adopted for our neural network was the LeNet-5 model, which was first introduced by LeCun et al. in 1998 for digit recognition tasks [1]. We use Adam optimizer XXX Utilizing this architecture, our model achieved a validation accuracy of 0.89. Although this performance is quite respectable, we aimed to enhance this accuracy further.
 
Figure 1

	Improvements after LeNet
	Adding layers has the effect of increasing accuracy on the test set. However, as shown in [x], increasing layers beyond a certain point reduces test and validation accuracy.
	Different activation functions like ReLU and leaky ReLU 
	Dropout to avoid overfitting in models with large fully connected layers.
	Regularization terms ect. L1, L2 normalization
	Data augmentation
	Skip connections (famously used in ResNet)
	Adam optimizer
Dataset:
	Project-Traffic Sign Classifier.pdf => P.22
Tools: 
	Python 3.10
	Jupyter Notebook 
Libraries: 
	matplotlib
	numpy
	cv2 (OpenCV)
	os
	math
	Tensorflow


III. Results
1. Adding layers:
To accomplish this, we decided to increase the depth of the model, following the seminal work of Simonyan and Zisserman, which demonstrates the effectiveness of deep architectures in visual recognition tasks [2]. However, upon implementing this change, we noticed a concerning pattern: while our model's training accuracy was very high (0.97), the validation accuracy was significantly lower. This discrepancy is a classic indication of overfitting, suggesting that our model was memorizing the training data rather than learning generalizable features.
 
Figure 2

2. Data augmentation:

Data augmentation is a strategy that allows us to significantly increase the diversity of data available for training models, without actually collecting new data. It is especially useful when dealing with image data. Techniques such as rotation, zooming, flipping, etc. can create different versions of the original images, thus contributing to improving the performance of the model.

Data augmentation techniques have been shown to reduce overfitting and improve model generalization. For example, a 2019 paper by Shorten and Khoshgoftaar titled "A survey on Image Data Augmentation for Deep Learning" (Journal of Big Data volume 6, Article number: 60) reviewed multiple studies and concluded that data augmentation can effectively reduce overfitting in deep learning models.

In Keras, we can apply data augmentation easily using the ImageDataGenerator class.
 
Figure 3


3. Dropout

Upon integrating dropout into our architecture, we noticed a substantial improvement in our model's performance. Our validation accuracy rose to 0.96.
 
Figure 4
Dropout, as proposed by Srivastava et al., is a technique that prevents overfitting by randomly setting a fraction of input units to 0 during training [3]. This encourages the model to learn robust representations, as it cannot rely on any single input unit. This has a similar effect to applying L2 regularization.
4. Early stopping:
As seen in figure 5, Training our data on too many epochs can still cause overfitting and degradation of model performance. We add an early stopping criterion to end model training early when loss increases for consecutive epochs. 
 
Figure 5

5. Choosing Activation functions
It is generally good to experiment with activation functions, as in different case, different functions tend to perform a bit differently. 
ReLU: f(x) = max(0,x)
Advantages:
	Computationally efficient: The function is simple to compute.
Reduced likelihood of vanishing gradient: This makes ReLU particularly useful for deep neural networks.
	Sparsity: ReLU's property of outputting actual zero (not just values close to zero) leads to sparsity, which can be a desirable property in some models.
	
Disadvantages:
	Dying ReLU problem: For inputs less than zero, the output is zero, so once a neuron gets negative it is unlikely for it to recover. This is known as the "dying ReLU" problem.
	Not zero-centered: The output is always positive, which might lead to optimization issues during the gradient descent process.
Tanh: 2 / (1 + exp(-2x)) - 1
Advantages:
	Zero-centered: Unlike ReLU, the output of tanh is zero-centered, which could help with the gradient descent optimization process.
	It can model negative inputs: Unlike ReLU, which can only output positive values, tanh can also model negative relationships.
	
Disadvantages:
	Computationally more expensive than ReLU.
	Vanishing gradient problem: For very large positive or negative inputs, the function saturates at -1 or 1, and the gradient is nearly zero. This can lead to slower convergence or the model not learning further.

Comparing the results from these two activation functions:
  
Figure 6



We can see that ReLU converges faster in our test. However, this test is not robust enough to prove the merits of one over the other. In our case, choosing ReLU allows us to perform well enough on validation data in minimal epochs.

There is also not visible overfitting in either case:
Tanh:                  ReLU:  

In the end, we choose Leaky ReLU to avoid the “dying ReLU problem” 

 
Figure 7
At this point we already achieve around 0.95 validation accuracy
 
6. Skip connections
ResNet[3] solves the problem of very deep neural networks perfroming worse by introducig skip connections. The core idea of ResNet is the introduction of the "identity shortcut connection"
  
Figure 8

that skips one or more layers.[x] we built the ResNet18 model from scratch by defining two types of blocks: identity blocks and convolutional blocks. They are then used to craft ResNet18 described as such in the original paper.
 
Our final highest test accuracy was achieved by ResNet:
 
7. Transfer Learning
Transfer learning is useful for finetuning a pre-trained model for a related task. However this is only particularly useful in scenarios where there is not enough labeled data to train a deep model from scratch. My conclusions from working with transfer learning on this particular dataset are that it might be more effective to train a model from scratch, specifically tailored to this task.
We apply transfer learning to our data set using a pretrained MobileNetV2. The process involves upscaling our data from 32x32 to 96x96, and then extracting high-features using  the pre-trained model. We then re-train the end fully-connected layers to output classification result for our 43 sign types.
Although this process only involves finetuning and upscaling, the training time(10 minutes)  to achieve comparable results to our smaller model trained from scratch(2 minutes) is much longer, thus negating the need to do so for this particular task.

IV. Discussion
A. Interpretation of results
By implementing these state-of-the-art methods for image classification, we saw major improvements in our model’s accuracy and robustness. Transfer learning was not incredibly useful for our dataset, however, since small models trained from scratch performs well on our dataset. This evidently tells us that we do not need large models that generalize well for such specific tasks. Instead, specialization in a certain task will often reduce model complexity and training/inference cost.

B. Future directions for improvement
The best result was achieved by the ResNet model. This is not surprising, since it is the newest model of the three. However, the size and recourses it takes to run and train the ResNet18 model is significantly larger than our custom model. In fact, the size of the ResNet model exceeds 100Mb, making larger versions of ResNet impractical to run on edge devices. If I was choosing models based on a tradeoff between size, accuracy, and cost to train, I would choose a “shallower” model just to reduce the memory it takes to run these models. 
One method that addresses this issue is the use of module neural networks[4].  
Figure 9
Instead of one large DNN, the task of classification is split into subtasks, effectively reducing the size of the loaded model. This has multiple benefits, like lower energy consumption and faster inference speeds. The hierarchy of classes and subclasses is a subproblem that can be solved by viewing these separate DNNs as modules. The output of the Parent DNN becomes the input to its children. The grouping of classes is achieved by predicting results using a pretrained DNN and then grouping classes with similar SoftMax outputs. To select size of individual modules the increase in accuracy/memory is compared to a set threshold value.
V. Conclusion
A. Summary of findings
Adding Layers: Increasing the depth of your model resulted in high training accuracy, but the validation accuracy was significantly lower, indicating overfitting – the model memorized the training data instead of learning generalizable patterns.
Data Augmentation: Utilized data augmentation techniques such as rotation, zooming, flipping, etc. to create variations of the original images. This method can reduce overfitting and improve model generalization by increasing the diversity of the training data.
Dropout: This change significantly improved your model's performance, with validation accuracy rising to 0.96.
Early Stopping: Added an early stopping criterion to end model training when loss increases for consecutive epochs. This approach helps to prevent overfitting and degradation of model performance from excessive training.
Choosing Activation Functions: Experimented with different activation functions, including ReLU and Tanh. ReLU converged faster in my tests, but to avoid the "dying ReLU problem", we chose Leaky ReLU.
Skip Connections (ResNet): Implemented ResNet18 from scratch to address the issue of performance degradation in very deep networks. The ResNet model achieved your highest test accuracy.
Transfer Learning: For this particular task, it might be more effective to train a model from scratch.

VI. References
[1]“Gradient-based learning applied to document recognition,” Gradient-based learning applied to document recognition | IEEE Journals & Magazine | IEEE Xplore. https://ieeexplore.ieee.org/document/726791
[2]N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” Dropout: A Simple Way to Prevent Neural Networks from Overfitting. https://jmlr.org/papers/v15/srivastava14a.html
[3]He, K. et al. (2015) Deep residual learning for image recognition, arXiv.org. Available at: https://arxiv.org/abs/1512.03385 
[4]A. Goel, S. Aghajanzadeh, and C. Tung, "Modular Neural Networks for Low-Power Image Classification on Embedded Devices," ACM Trans. Des. Autom. Electron. Syst., vol. 26, no. 1, Article 1, Oct. 2020, pp. 1-3512.

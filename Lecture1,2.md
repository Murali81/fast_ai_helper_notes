Sources : http://forums.fast.ai/t/precompute-true/7316/50

Other sources: https://medium.com/@apiltamang/case-study-a-world-class-image-classifier-for-dogs-and-cats-err-anything-9cf39ee4690e
               https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
Hello,

This is my understanding of precompute, unfreeze, etc. (and whatever is done jupyter notebooks after the 2 first lessons) :

1. # Setup your variables
PATH = "data/dogscats/"; arch=resnet34; sz = 224
bs=64 (batch size) is the default value in methods used below, so you don’t need to define it here.

2. # Setup your data augmentation (DA)
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
The DA of your training images will have an impact only if precompute=False in your new model (learn) but if you define your DA now (thanks to aug_tfms), you will not need to take care of it after (ie, when you will run learn.precompute=False, cf point 6).

3. # Format your data
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
At this point, your data (images) are formatted according to your pre-trained model (arch) and preferences (sz, DA, zoom…), and they are ready-to-be used.

4. # Setup your new neural network (NN)
learn = ConvLearner.pretrained(arch, data, precompute=True)
The pretrained method creates your new NN from the arch model :

* by keeping all layers but the last one (ie, the output one which gives probabilities within 1000 classes ImageNet)
* which is remplaced by adding few layers (@jeremy will give details later in the course I think) that end with an output layer which gives probabilities within 2 classes (dogs, cats).

At its creation (ie, when you run the code above : learn = ...) and by default, the new NN freezes the first layers (the ones from arch) and downloads the pre-trained weights of arch.
More, precompute=False by default. Therefore, you must precise precompute=True if you want to change the default behavior.

What does precompute=True ? It tells your new NN learn to process only one time your data (images) through the arch model (but its last layer that was removed) using its pre-trained weights. That’s what we mean by the expression “compute the activations”. This transformation by activation of your data is done only one time and now the new values of your data can be used as inputs of the last layers of your new NN that you are about to train (cf point 5).

Note 1 : even if you have put on the data augmentation (cf aug_tfms in point 2), this has no impact when precompute=True as the activation of your data (images) is computed only one time. So, at each new epoch used in the training, the values used as inputs of the last layers of your new NN are the ones computed at the first epoch.

Note 2 : there is no obligation to set precompute=True but the training of your new NN will be faster as your data (images) are processed only one time through the first layers. Therefore, it is interessing when you start your project.

5. # Train the last layers of your new NN
learn.fit(0.01,1)
Through lr_find(), you choose the best learning rate and then, train your NN using the fit method (use 1 to 3 epochs). At this point (precompute=True and first layers frozen), only the last layers of your new NN will be trained (ie, their weights will be updated in order to minimize the loss of the model).

6. # Improve the weights of your last layers by data augmentation and SGDR
learn.precompute=False
The more data you have, the better model you will get. Set precompute=False and then, at each new epoch used in the training of your new NN, the activation of your augmented data (cf point 2) will be computed. As well, you should use the stochastic gradient descent with restarts (SGDR) at this point.

7. # Improve your new NN (all layers)
learn.unfreeze()
At this point, only your last layers have been trained. Then, you should trained now all the layers of your new NN together. This is done by setting unfreeze the first layers of learn.

Note : before to train again your model (using fit), you should use the lr_find() method again in order to select the best learning rate of your NN with all layers unfrozen. As well, you should use the differential learning rates and cycle_mult parameter.

8. # Final steps : increase sz and move to a better pre-trained model

See the https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1-rxt50.ipynb28 jupyter notebook

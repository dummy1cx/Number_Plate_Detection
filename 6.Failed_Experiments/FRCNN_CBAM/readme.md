
#
The code was referenced from week_3 lectures.
#
The experiment was done over the architecture of FRCNN. CBAM was integrated into the backbone of architecture with a objective of higher accuracy. But it was noticed that the model was not learning anything. It was observed training loss decreased substantially and the model is performing nice in predicting bounding box in the training data but failed to do so in unseen data.
#
The training was stopped because of the poor performance of the model.
#
We can comeback to the model with implementions like Hyper-parameter tuning and with a relatively bigger dataset. As the model was trained with only 400 images (approx)
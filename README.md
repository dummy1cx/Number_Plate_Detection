# Number_Plate_Detection
![Number Plate Detection](gif.gif)

#### Introduction
Automatic Number Plate Detection is now an integral part of modern transport infrastructure because of its various applications including collecting tolls and taxes, stolen vehicle identification, parking management and many more. Classical CNN based models have their own limitations in processing blurry and unclear vision which led to transition in modern deep learning models for highest accuracy in lowest processing time due to which now processing a 60 fps video is possible irrespective of lighting condition and angle of the images with the help of modern deep learning techniques. The ultimate goal is to predict the bounding box surrounding the number plate and read the text using OCR whenever the model sees any images or videos consisting of any vehicles.

#### Data Pre-Processing
The dataset employed for the experiments was sourced from the Car License Plate Detection dataset available on Kaggle. It comprises approximately 433 images of vehicles with annotated number plates. The bounding box annotations are provided in the Pascal VOC format with XML annotations, which facilitates standardized evaluation and model training. For YOLO training the annotations are converted into YOLO supported file format. In the Pascal VOC format, bounding box annotations are specified using the coordinates of the top-left and bottom-right corners, represented as Xmin, Xmax, Ymin, Ymax. However, the YOLO (You Only Look Once) object detection framework requires annotations in a different format, where each bounding box is defined by its center coordinates (XCentre, YCentre,) along with its Width and Height.  Consequently, a conversion process is necessary to reformat the annotations from Pascal VOC to YOLO format. This involves computing the center point of the bounding box and its dimensions based on the original corner coordinates. 

Formula for YOLO Annotations //// 
XCentre = (Xmin + Xmax) / 2 //// 
YCentre = (Ymin + Ymax) / 2 //// 
Height = (Ymax - Ymin) ////
Width = (Xmax - Xmin)

#### Architecture
The models will be discussed in the essay as follows :

Baseline model 1 ⇔ YOLOv8 by Ultralytics

Custom YOLO with CBAM attention

Baseline model 2 ⇔ Faster R-CNN

Custom Faster R-CNN

#### Model
To achieve optimal number plate detection, this project initially employed YOLOv8 by Ultralytics and Faster R-CNN with a pre-trained ResNet-50 backbone as baseline models. The primary objective was to surpass the performance of these models by designing a modified architecture with enhanced detection capabilities. To this end, a series of experiments were conducted. These included the integration of Convolutional Block Attention Module (CBAM) mechanisms into YOLOv4, and the replacement of the YOLOv4 detection head with a novel YOLOv11 detection head. Additional experimentation involved incorporating spatial and channel attention modules into the neck of the architecture alongside the Feature Pyramid Network (FPN), although this yielded only marginal improvements in detection accuracy.
Subsequently, the project explored the integration of vision transformers, specifically the Swin Transformer, with Convolutional Neural Networks (CNNs) to enhance feature extraction in the backbone. After extensive experimentation, two of the most effective custom architectures were identified. These will be discussed in detail in the coursework, along with a comprehensive analysis of their performance metrics and accuracy results.

#### Results

#### Conclusion
Throughout this project “Automatic Number Plate Recognition”, I gained a deeper understanding about advanced object detection techniques like YOLO and Faster RCNN. I improved my practical skills  by developing models based on previous architecture. From my study I learnt that modern pertained backbones are pre trained with millions of images and implementing attention mechanisms do not guarantee improvement unless the data demands extra focus. HIgh accuracy can be achieved from high quality dataset and careful tuning of hyper parameters.In future my experiments would extend with harder datasets like blurry and unclear images and improving the quality using SRGAN like architecture. I would then compare the model with the attention mechanism to validate my findings from this project. Again implementing an advanced OCR might do well for text extraction and detection.
#### Reference
All the sources have been referenced in the written report.

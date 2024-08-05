---
layout: post
title: Object Detection on Argoversehd Dataset - Exploring YOLOv8 Models
date: 2023-06-07 10:14:00-0400
description: Object detection is performed on ArgoverseHD-Dataset
tags: computer_vision image_processing AI ML 
categories: Object_detection Yolov8 #sample-posts toc sidebar
giscus_comments: true
thumbnail: assets/img/blog2/Thumbnail.gif
related_posts: true

authors:
  - name: Anirudh Iyengar
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:  
     name: Post grad student,Arizona State Univeristy
  # - name: Boris Podolsky
  #   url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
  #   affiliations:
  #     name: IAS, Princeton
  # - name: Nathan Rosen
  #   url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
  #   affiliations:
  #     name: IAS, Princeton

bibliography: cadpepapers.bib

toc:
  sidebar: left
---
## Object detection using YoloV8 model
### Introduction
Object detection, a subfield of computer vision, plays a crucial role in various domains, including driving. It enables the automated identification and localization of objects within images or videos, providing valuable insights and aiding in decision-making processes. While there are numerous models available for object detection, this blog focuses on YOLOv8, a state-of-the-art approach renowned for its accuracy and efficiency. In this experiment, we delve into the application of YOLOv8 models on the Argoversehd dataset, comparing the results obtained and exploring how YOLOv8 performs in this context.

In the subsequent sections of this blog, we will dive deeper into the experimental setup, highlighting the key steps involved in training and evaluating the YOLOv8s and YOLOv8m models on the Argoversehd dataset. We will discuss the training process, including data preparation, model configuration, and hyperparameter tuning. Additionally, we will present the evaluation results and analyze how YOLOv8 performs compared to the original dataset. This comparative analysis will shed light on the strengths and limitations of YOLOv8 in the context of object detection on the Argoversehd dataset.

Stay tuned as we embark on this exciting journey into the realm of object detection using YOLOv8 models. Through this experiment, we hope to gain insights that can contribute to the advancement of object detection techniques in the autonomus driving domain, fostering innovations that can benefit researchers alike.[^1]
<!-- <sup id="fnref:1"><a href="#fn:1" class="footnote-ref">1</a></sup>  -->

### Argoverse Dataset
#### Preparation for YOLOv8 Object Detection

The <a href="https://mtli.github.io/streaming/"><b>Argoverse dataset</b></a>, which forms the basis of our object detection experiment using YOLOv8 models, consists of a total of 66,954 images. The dataset is divided into three subsets: training, validation, and testing, with 39,384, 12,507, and 15,063 images, respectively. The training and validation subsets contain annotations in the COCO format, while the testing subset lacks ground truth annotations. In the absence of annotations, we will utilize the trained YOLOv8 models to predict and detect objects within the test images.

The Argoverse dataset encompasses eight classes of objects, namely: "person," "bicycle," "car," "motorcycle," "bus," "truck," "traffic_light," and "stop_sign." These classes represent common objects typically found in vechile driving contexts.

To prepare the dataset for YOLOv8, a specific directory structure is required.
```yml
root_data
├── train
│   ├── images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val
│   ├── images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── test
    └── images
        ├── image1.jpg
        ├── image2.jpg
        └── ...

```
To meet the YOLO format requirements, the annotations need to be converted from the COCO format to the YOLO format. Each linein the labels text file represents a single object annotation with its corresponding class ID, normalized coordinates, and dimensions.

```yml
Image Label Text file:
class_id x_center y_center width height
class_id x_center y_center width height
class_id x_center y_center width height
...

```
Here is a sneak peek of the code used to convert the annotations to the YOLO format:
```yml

def convert_annotations_to_yolo_format(data, file_names, output_path):
    def get_img(filename):
        for img in data['images']:
            if img['file_name'] == filename:
                return img

    def get_img_ann(image_id):
        img_ann = []
        isFound = False
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                img_ann.append(ann)
                isFound = True
        if isFound:
            return img_ann
        else:
            return None

    count = 0

    for filename in file_names:
        # Extracting image
        img = get_img(filename)
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        # Get Annotations for this image
        img_ann = get_img_ann(img_id)
        fname = filename.split(".")[0]
        if img_ann:
            # Opening file for the current image
            file_object = open(f"{output_path}/{fname}.txt", "a")
        if img_ann is not None:
            for ann in img_ann:
                current_category = ann['category_id']  # As YOLO format labels start from 0
                current_bbox = ann['bbox']
                x = current_bbox[0]
                y = current_bbox[1]
                w = current_bbox[2]
                h = current_bbox[3]

                # Finding midpoints
                x_centre = (x + (x+w))/2
                y_centre = (y + (y+h))/2

                # Normalization
                x_centre = x_centre / img_w
                y_centre = y_centre / img_h
                w = w / img_w
                h = h / img_h

                # Limiting up to a fixed number of decimal places
                x_centre = format(x_centre, '.6f')
                y_centre = format(y_centre, '.6f')
                w = format(w, '.6f')
                h = format(h, '.6f')

                # Writing the current object
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

        file_object.close()
        count += 1
```
After the conversion, the labels are saved as individual text files in the "labels" folder, corresponding to each image.

With the dataset now prepared in the YOLOv8 format, we can proceed to train and evaluate the YOLOv8s and YOLOv8m models on the Agroverse dataset.

### YOLOv8 Architecture
YOLOv8 is an evolution of the YOLO (You Only Look Once) family of models, designed for efficient and accurate object detection.

One significant update in YOLOv8 is its transition to anchor-free detection. Traditional object detection models often rely on predefined anchor boxes of different scales and aspect ratios to detect objects at various sizes. However, YOLOv8 takes a different approach by predicting the center of an object directly, rather than the offset from predefined anchor boxes.
<div class="row mt-3">
        {% include figure.liquid path="assets/img/blog2/Anchor-free.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>
<div class="caption">
    Figure 1: Difference between Anchor-free and Anchor based. <a herf= " https://www.nature.com/articles/s41598-021-02095-4"> <em> Image Source</em> </a>
</div>

This anchor-free approach brings several advantages. Firstly, it simplifies the model architecture by removing the need for anchor boxes and associated calculations. This leads to a more streamlined and efficient network. Additionally, anchor-free detection allows for better localization accuracy, as the model directly predicts the object center with high precision.

To visualize the YOLOv8 architecture and its anchor-free detection, we can refer to a detailed diagram created by GitHub user RangeKing (shown below). The diagram provides a comprehensive overview of the network's structure and the flow of information through different layers.


<div class="row mt-3">
        {% include figure.liquid path="assets/img/blog2/yolov8.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>
<div class="caption">
    Figure 2: YOLOv8 Architecture.
</div>

**visualisation made by GitHub user** <a href = "https://github.com/RangeKing" ><b>RangeKing</b></a>

By adopting anchor-free detection, YOLOv8 enhances object detection performance. 

### Training YOLOv8 on Agroverse Dataset
To train YOLOv8 on the Argoverse dataset, we need to create a `data.yaml` file and install the necessary dependencies. Here's a step-by-step guide to training YOLOv8 on the Argoverse dataset:

##### **Create the `data.yaml` File**: 

Before training, we need to create a `data.yaml` file to specify the dataset's configuration. The structure of the `data.yaml` file is as follows:

```yml
path: /your/root/path
train: root/train/images/
val: root/val/images/
nc: number_of_classes
names: [class1, class2, ..., classN]
```

Ensure that you replace `/your/root/path` with the actual root path of your dataset, `root/train/images/` with the path to the training images folder, `root/val/images/` with the path to the validation images folder, `number_of_classes` with the total number of classes in your dataset, and `[class1, class2, ..., classN]` with a list of the class names in string format.

##### **Install Dependencies**: Install the required dependencies by running the following command:

```python
!pip install ultralytics
```

##### **Import YOLO and Load the Model**: Import the `YOLO` class from the `ultralytics` package and load the YOLOv8 model using the desired `.pt` file:

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
```

The `YOLO` class from `ultralytics` automatically downloads the required YOLOv8 models, such as `yolov8s` or `yolov8m`, based on the specified `.pt` file.

##### **Start Training**: Begin the training process by calling the `train` method on the `model` object with appropriate arguments. Here's an example configuration:

```python
output = model.train(
   data='Argoverse.yaml',
   imgsz=512,
   epochs=10,
   batch=8,
   save=True,
   name='yolov8m_custom',
   val=True,
   project='yolov8m_custom_Argoverse',
   save_period=2
)
```

In this example, we specify the `data` parameter as `'Argoverse.yaml'` to use the created `data.yaml` file. Adjust the other parameters such as `imgsz` (image size), `epochs` (number of training epochs), `batch` (batch size), `save` (whether to save checkpoints), `name` (name for the trained model), `val` (whether to evaluate on the validation set), `project` (project name for logging), and `save_period` (number of epochs between saving checkpoints) according to your requirements.

##### **Monitor Training Progress**: 
During training, the YOLO model will provide updates on the training loss, bounding box loss, mean Average Precision (mAP), etc.

For more detailed information and additional training options, refer to the <a herf= "https://docs.ultralytics.com/modes/train/"><b>YOLOv5 Train Mode Documentation</b></a> provided by Ultralytics.

### Testing the Trained YOLOv8 Model
After training the YOLOv8 model on the Argoverse dataset, it's time to evaluate its performance on the test data. In this section, we will test the best trained YOLOv8s and YOLOv8m models on the test dataset.

Firstly, the test data for Argoverse consists of individual images. To provide a more comprehensive evaluation, I converted 2000 frames of the test data into a video at 24 frames per second (fps). This video allows for a sequential analysis of the model's object detection capabilities. Also predicted on whole test data.

Here's an example configaration :

 ```python
 model.predict('Yolov8/test_video.mp4', save=True, conf=0.5) # You Can also add path to your images
 ```
 
Below are the videos showcasing original and the testing results of the YOLOv8s and YOLOv8m models on the test data:
**Test Video**

<div class="row mt-3">
        <iframe width="1002" height="626" src="https://www.youtube.com/embed/SeRUThVhlc4" title="Test Video for testing YoloV8 model" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

<div class="caption">
    <b> Original Test video.</b>
</div>

**YOLOv8s Predicted Video**

<div class="row mt-3">
        <iframe width="1002" height="626" src="https://www.youtube.com/embed/NMq17lLEHEw" title="Prediction of YoloV8s model" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

<div class="caption">
    <b> YOLOV8s Prediction.</b>
</div>

**YOLOv8m Predicted Video:**

<div class="row mt-3">
        <iframe width="1002" height="626" src="https://www.youtube.com/embed/2_2clDwQSb0" title="Prediction of YOLOv8m model" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

<div class="caption">
   <b> YOLOV8m Prediction.</b>
</div>

By visually examining the test videos, we can observe how the YOLOv8 models detect and classify objects in the Argoverse test dataset. The models' performance will be evident in their ability to accurately identify and localize objects of interest, such as people, bicycles, cars, motorcycles, buses, trucks, traffic lights, and stop signs.

The models will output bounding boxes around the detected objects, along with their corresponding class labels and confidence scores.

### Analyzing the Test Results

After testing the YOLOv8 models on the Argoversehd dataset and evaluating the results, it is important to conduct a thorough analysis to gain insights into the performance of the models. This analysis involves both visual inspection and the use of quantitative metrics to assess the models' effectiveness in object detection tasks.

#### Visual Inspection:
Upon visually inspecting the test results, it becomes evident that the YOLOv8 models show promising performance in detecting and localizing objects. However, there are areas where the models exhibit limitations. For example, the models incorrectly identify certain objects as trucks and miss some instances of stop signs. These observations suggest that further improvements can be made by refining the training process and incorporating additional data.

<div class="row mt-3">
        {% include figure.liquid path="assets/img/blog2/results.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>

#### Quantitative Metrics:
The mean Average Precision (mAP) is a widely used metric for evaluating object detection models. The mAP measures the accuracy of object localization and classification. In the case of the YOLOv8 models trained on the Argoversehd dataset, the highest achieved mAP is 0.40, indicating good performance for certain instances. However, the average mAP typically falls within the range of 0.24 to 0.35. This implies that there is room for improvement in terms of the models' overall accuracy and precision.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/blog2/F1_curve.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/blog2/PR_curve.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### Confusion Matrix:
A confusion matrix provides a detailed breakdown of the model's performance across different object classes. By analyzing the confusion matrix, we can identify specific areas where the YOLOv8 models excel and areas where they struggle. In the case of the Argoversehd dataset, the YOLO models face challenges in accurately detecting small objects and occasionally misclassifying certain objects. To address these limitations, it is advisable to consider strategies such as increasing the amount of training data and conducting further model optimization.

<div class="row mt-3">
        {% include figure.liquid path="assets/img/blog2/confusion_matrix.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>

#### Improving Model Performance:
Based on the analysis of the test results, it is clear that there is room for improvement in the YOLOv8 models' performance on the Argoversehd dataset. By implementing different strategies and iteratively training and evaluating the YOLOv8 models, it is possible to improve their object detection accuracy and address the specific challenges observed during testing on the Argoversehd dataset.

***Footnote***:

[^1]: ***Beginner's Work and Request for Understanding*** - *Please note that this blog and the work presented herein are the efforts of a beginner in the field of image processing. While every attempt has been made to ensure accuracy and provide valuable insights, there may be certain limitations or areas for improvement. If any inconveniences or shortcomings are encountered, I kindly request your understanding and forgiveness. This blog serves as a starting point for exploring the fascinating world of Image processing and computer vision, and I am eager to learn and grow from this experience. Your feedback and suggestions are greatly appreciated as they will contribute to my growth as a learner and researcher. Thank you for your support and understanding.*

<!-- <div id="footnotes">
  <ol>
    <li id="fn:1">
      <p>Beginner's Work and Request for Understanding<br>
Please note that this blog and the work presented herein are the efforts of a beginner in the field of image processing. While every attempt has been made to ensure accuracy and provide valuable insights, there may be certain limitations or areas for improvement. If any inconveniences or shortcomings are encountered, I kindly request your understanding and forgiveness. This blog serves as a starting point for exploring the fascinating world of Image processing and computer vision, and I am eager to learn and grow from this experience. Your feedback and suggestions are greatly appreciated as they will contribute to my growth as a learner and researcher. Thank you for your support and understanding. <a href="#fnref:1" class="footnote-backref">↩</a></p>
    </li>
  </ol>
</div> -->


<!-- To add a table of contents to a post as a sidebar, simply add
```yml
toc:
  sidebar: left
```
to the front matter of the post. The table of contents will be automatically generated from the headings in the post. If you wish to display the sidebar to the right, simply change `left` to `right`.

### Example of Sub-Heading 1

Jean shorts raw denim Vice normcore, art party High Life PBR skateboard stumptown vinyl kitsch. Four loko meh 8-bit, tousled banh mi tilde forage Schlitz dreamcatcher twee 3 wolf moon. Chambray asymmetrical paleo salvia, sartorial umami four loko master cleanse drinking vinegar brunch. <a href="https://www.pinterest.com">Pinterest</a> DIY authentic Schlitz, hoodie Intelligentsia butcher trust fund brunch shabby chic Kickstarter forage flexitarian. Direct trade <a href="https://en.wikipedia.org/wiki/Cold-pressed_juice">cold-pressed</a> meggings stumptown plaid, pop-up taxidermy. Hoodie XOXO fingerstache scenester Echo Park. Plaid ugh Wes Anderson, freegan pug selvage fanny pack leggings pickled food truck DIY irony Banksy.

### Example of another Sub-Heading 1

Jean shorts raw denim Vice normcore, art party High Life PBR skateboard stumptown vinyl kitsch. Four loko meh 8-bit, tousled banh mi tilde forage Schlitz dreamcatcher twee 3 wolf moon. Chambray asymmetrical paleo salvia, sartorial umami four loko master cleanse drinking vinegar brunch. <a href="https://www.pinterest.com">Pinterest</a> DIY authentic Schlitz, hoodie Intelligentsia butcher trust fund brunch shabby chic Kickstarter forage flexitarian. Direct trade <a href="https://en.wikipedia.org/wiki/Cold-pressed_juice">cold-pressed</a> meggings stumptown plaid, pop-up taxidermy. Hoodie XOXO fingerstache scenester Echo Park. Plaid ugh Wes Anderson, freegan pug selvage fanny pack leggings pickled food truck DIY irony Banksy.

## Customizing Your Table of Contents
{:data-toc-text="Customizing"}

If you want to learn more about how to customize the table of contents of your sidebar, you can check the [bootstrap-toc](https://afeld.github.io/bootstrap-toc/) documentation. Notice that you can even customize the text of the heading that will be displayed on the sidebar.

### Example of Sub-Heading 2

Jean shorts raw denim Vice normcore, art party High Life PBR skateboard stumptown vinyl kitsch. Four loko meh 8-bit, tousled banh mi tilde forage Schlitz dreamcatcher twee 3 wolf moon. Chambray asymmetrical paleo salvia, sartorial umami four loko master cleanse drinking vinegar brunch. <a href="https://www.pinterest.com">Pinterest</a> DIY authentic Schlitz, hoodie Intelligentsia butcher trust fund brunch shabby chic Kickstarter forage flexitarian. Direct trade <a href="https://en.wikipedia.org/wiki/Cold-pressed_juice">cold-pressed</a> meggings stumptown plaid, pop-up taxidermy. Hoodie XOXO fingerstache scenester Echo Park. Plaid ugh Wes Anderson, freegan pug selvage fanny pack leggings pickled food truck DIY irony Banksy.

### Example of another Sub-Heading 2

Jean shorts raw denim Vice normcore, art party High Life PBR skateboard stumptown vinyl kitsch. Four loko meh 8-bit, tousled banh mi tilde forage Schlitz dreamcatcher twee 3 wolf moon. Chambray asymmetrical paleo salvia, sartorial umami four loko master cleanse drinking vinegar brunch. <a href="https://www.pinterest.com">Pinterest</a> DIY authentic Schlitz, hoodie Intelligentsia butcher trust fund brunch shabby chic Kickstarter forage flexitarian. Direct trade <a href="https://en.wikipedia.org/wiki/Cold-pressed_juice">cold-pressed</a> meggings stumptown plaid, pop-up taxidermy. Hoodie XOXO fingerstache scenester Echo Park. Plaid ugh Wes Anderson, freegan pug selvage fanny pack leggings pickled food truck DIY irony Banksy. -->

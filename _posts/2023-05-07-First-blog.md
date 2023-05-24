---
layout: post
title:  Segmentation on CAD-PE Dataset. 
date: 2023-05-07 14:00:00
description: The segmentation is performed on CAD-PE dataset
tags: computer_vision image_processing AI ML Medical_images
# categories: sample-posts
thumbnail: assets/img/blog1/cadpe.gif
---
### CAD-PE Segmentation: Unveiling Insights

Welcome to our blog, where we delve into the fascinating world of segmentation. The segmentation is performed on __CAD-PE dataset__.
In the realm of `"computer-aided design(CAD)"`, the precision and efficent segementation plays a pivotal role.<br>

In this blog series, we will embark on an exciting journey to understand the challenges and intricacies of segmenting CAD-PE data. Whether you are a beginner or an experienced practitioner, we aim to provide valuable insights and practical guidance to enhance your understanding and proficiency in CAD-PE segmentation. Throught this blog, we will discuss various aspects of segementation,including data preprocessing, feature extraction, and model architectures. Moreover, we will dive into the evaluation metrics commonly used in assessing the performance of segmentation algorithms.

Join me as we unravel the complexities of segmentation on CAD-PE dataset, empowering you to leverage this knowledge in your research, industry projects, or even personal endeavours. So, fasten your seatbelts and get ready to explore the world of segmentation like never before!!.Let's unlock the hidden potential within Deep learning models and unleash their power.<br>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog1/cadpe.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog1/cadpemask.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 1: Unveiling the Hidden Layers: A GIF showcasing the CAD image and its corresponding ground truth mask.
</div>

### CAD-PE Dataset
    The first step involved is exploring the dataset. The dataset involves __91 patients CT scans__. Each CT scan consists of some around 400 to 500 slices on average. Dividing the CT scans of the 91 patients into individual slices. This process allowed us to extract __41,256 slices__ in total, which will serve as the foundation for our segmentation endeavors.

    Each slice within the CAD-PE dataset represents a two-dimensional image capturing a specific cross-section of the patients' anatomy. These slices provide crucial insights into the internal structures and organs, enabling medical professionals and researchers to diagnose and study various conditions and diseases.

### Building the Segmentation Dataset: Slice-Level Segmentation
    In order to perform slice-level segmentation on the CAD-PE dataset, we will create a custom dataset named "segmentation_dataset". This dataset will serve as the foundation for training and evaluating our segmentation algorithms.

    To begin, we will divide the available data randomly into three sets: training, validation, and testing. The training set will contain 80% of the data, while the validation and testing sets will each consist of 10% of the data. This division ensures a balanced distribution of slices across the different sets, enabling us to train and assess the performance of our models effectively.

    To handle the dataset efficiently, we will utilize filepaths to access the slices. Each slice within the CAD-PE dataset will be normalized to have pixel values ranging between 0 and 1. This normalization step ensures consistency and facilitates optimal model performance during training.

    Moreover, the input images and corresponding segmentation masks will be processed to adhere to the requirements of slice-level segmentation. The input images will be converted into single-channel representations, while the segmentation masks will be transformed into binary masks, consisting of only 0s and 1s. These binary masks serve as ground truth annotations for the presence or absence of the target structures within the slices.

    To facilitate seamless integration with deep learning frameworks, such as PyTorch, the slices will be transformed into tensors.

    The snippet code provided below showcases a high-level implementation for building the segmentation_dataset:

    ```html
    <div class="row justify-content-sm-center">
        <div class="col-sm-8 mt-3 mt-md-0">
            {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm-4 mt-3 mt-md-0">
            {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    ```


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/8.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/10.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

The rest of the images in this post are all zoomable, arranged into different mini-galleries.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/12.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/7.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

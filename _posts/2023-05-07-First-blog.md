---
layout: post
title:  Segmentation on CAD-PE Dataset. 
date: 2023-05-07 14:00:00
description: the Story of lessons
tags: Medical images
# categories: sample-posts
thumbnail: assets/img/blog1/cadpe.gif
---
### CAD-PE Segmentation: Unveiling Insights

Welcome to our blog, where we delve into the fascinating world of segmentation. The segmentation is performed on CAD-PE dataset.
In the realm of "computer-aided design(CAD)", the precision and efficent segementation plays a pivotal role.<br>

In this blog series, we will embark on an exciting journey to understand the challenges and intricacies of segmenting CAD-PE data. Whether you are a beginner or an experienced practitioner, we aim to provide valuable insights and practical guidance to enhance your understanding and proficiency in CAD-PE segmentation. Throught this blog, we will discuss various aspects of segementation,including data preprocessing, feature extraction, and model architectures. Moreover, we will dive into the evaluation metrics commonly used in assessing the performance of segmentation algorithms.

Join me as we unravel the complexities of segmentation on CAD-PE dataset, empowering you to leverage this knowledge in your research, industry projects, or even personal endeavours. So, fasten your seatbelts and get ready to explore the world of segmentation like never before!!.Let's unlock the hidden potential within Deep learning models and unleash their power.<br>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog1/cadpe.gif" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog1/cadpemask.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Unveiling the Hidden Layers: A GIF showcasing the CAD image and its corresponding ground truth mask.
</div>

#### CAD-PE Dataset
 
 The first step involved is exploring the dataset. The dataset involves 91 patients CT scans. Each CT scan consists of some around 400 to 500 slices on average. Dividing the CT scans of the 91 patients into individual slices. This process allowed us to extract 41,256 slices in total, which will serve as the foundation for our segmentation endeavors.

Each slice within the CAD-PE dataset represents a two-dimensional image capturing a specific cross-section of the patients' anatomy. These slices provide crucial insights into the internal structures and organs, enabling medical professionals and researchers to diagnose and study various conditions and diseases.

Images can be made zoomable.
Simply add `data-zoomable` to `<img>` tags that you want to make zoomable.

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

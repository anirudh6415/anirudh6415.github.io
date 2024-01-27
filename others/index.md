---
layout: page
title: Quest
description: A peek into my life experiences
importance: 1
category: Personal
---
My journey through this life has been unique due to my diverse interests and passion. I love exploring the world through the eyes of people from different backgrounds. I enjoy spending time with strangers in the city during my travels and exploring the world in their eyes. This has encouraged me to look at a situation with different perspectives and depth. I realized that the difference in perception of reality among different people is the most fascinating thing and it helped me achieve a positive outlook even in times of despair. 

The challenges posed by the COVID pandemic became an opportunity for me to tap into my entrepreneurial spirit. Stepping into the shoes of my family's business, which has thrived for five decades, was no small feat, especially as I navigated the expectations of my father's loyal customers. I realized that success seldom comes easily. But it is possible to not only get beyond the obstacles but also learn from them if one has the right attitude. Leveraging the power of data analysis, I examined sales data from the past two years and identified the most popular products. With a focus on freshness and quality, I developed a slogan, "Make Fresh Sell Fresh," and implemented new packaging techniques. This organized approach not only helped me gain better feedback from the customer but also my team.

<div class="caption">
    Memories in Pixels Captured by Moments 
</div>

<div class="row mt-3">
        {% for i in range(1, 28) %}
            <div class="col-sm-4 mt-3 mt-md-0">
                {% capture img_path %}assets/img/others/IMG-20240127-WA00{{ "%02d" | format(i) }}.jpg{% endcapture %}
                {% include figure.html path=img_path title="Example Image" class="img-fluid rounded z-depth-1" zoomable=True %}
            </div>
        {% endfor %}
</div>

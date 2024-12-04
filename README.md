# LLM experiments

## datasets

### stress-detection-from-social-media-articles

DS introduced in "Stress Detection from Social Media Articles: New Dataset Benchmark and Analytical Study" (https://ieeexplore.ieee.org/document/9892889 / https://sentic.net/stress-detection-from-social-media.pdf)
Datasets collected from reddit and twitter APIs, automatically annotated using a RoBERTa finetuned on the *Twitter Emotion dataset*, with a probability threshold of 90%.

<details>
<summary>Click to expand</summary>

Amonst the 6 classes of the twtter emotions DS (sadness, joy,
love, anger, fear and surprise) - sadness, anger and fear are chosen to represent positive Stress.  
The authors claim to have validated the performance of automated annotations using gold subsets (manually annotated subsets of 234 examples for Reddit and 120 for twitter), they report an average accuracy of 94% across datasets 

</details>
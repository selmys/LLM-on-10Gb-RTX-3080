# LLM-on-10Gb-RTX-3080
I wanted to build a complete LLM, with tinystories.txt as my data set, from pre-training to fine-tuning, on my home PC. Of course, I used AI to help me but I still ran into many problems, most dealt with my limited hardware. With only 10Gb of VRAM, I had to try dozens of AI suggestions until I maximized the use of my graphics card. 

In the end I managed to get a 124M parameter tiny LLM. My pre-training code used 9Gb VRAM and ran for almost 30 hours - over 110,000 iterations. The inference code produced good results.

For fine-tuning I generated about 13K questions and answers using vllm with a down-loaded model. I then asked AI to make a web page so I could test my model. The web-app python scripts worked really well and the results of my questions were quite good. 

The data files for this project are quite large so you can find them here: https://lotuspond.ca/~selmys/tinyLLM/

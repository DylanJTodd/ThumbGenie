# ThumbGenie

Thumbgenie will allow you to create custom YouTube thumbnails to use on your video, taking the title and category as inputs.

You can see some examples *below:*


![thumbgenie demo](images/Thumbgenie-Demo.png?raw=true)

## **How to use:**

 - **[Manually:](https://github.com/DylanJTodd/ThumbGenie/main/README.md#manually)**
 - **[Through Google Colab](https://github.com/DylanJTodd/ThumbGenie/main/README.md#through-google-colab)**


## Manually

 1. First, clone the repository locally

	    git clone https://github.com/DylanJTodd/ThumbGenie

 2.  Change into the directory created

		 cd ./ThumbGenie/

 3.  Next, pip install requirements.txt (Note, need python version 3.10+ )

		 pip install -r requirements.txt

 4.  If you want to train, then run preprocessing.py, followed by training.py. Ensure to tune the parameters such as epochs, batch size, resolution, etc, to your liking

		 python preprocessing.py
		 python training.py
	    
 5.  Run thumbgenie.py to start generating custom images. Please note, this file does come with a pretrained model, however due to limited resources, it was not trained thoroughly. I would recommend anyone with the ability to do so, to train their own model.

		 python thumbgenie.py

## Through Google Colab

1. Open this [notebook](https://colab.research.google.com/drive/18TOMosq6KhXi_oArdFLjiHXmNKgI6xFX?usp=sharing)

2. Make a personal copy, by going to the top left, through **File > Save a copy in Drive**

3. Follow the instructions on the notebook.

 

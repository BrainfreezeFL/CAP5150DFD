# CAP5150DFD

Please use the files in final code for the final project, the other files are from the checkin point. The models and the testing data you will need are too big to attach here, so I will link them using google drive here:

https://drive.google.com/drive/folders/1HStct03TeDADHe86P9LK5gRdcyVSoqFN?usp=sharing

It should be accessible to anyone with the link, but if it is not, please message me and I will add you directly. 

I ran the models using a docker container with the tensorflow image you can get from their site. running the retrieve_model_colab.py script with all of the models and training data within the same directory should work. I modified the file loaders to look for them there, but it was designed to fetch them in other locations on my system. If I messed up, this can be resolved by modifying where it fetches the testing information on line 56. If it can't find the models this starts around line 169.

The bulk of the rest of the code was modified to generate information for a potential fix for the DFD vulnerability that didn't pan out. I will address this in my project report though. They should still be runnable if you grab the original datasets from here though : 
https://drive.google.com/drive/folders/1kxjqlaWWJbMW56E3kbXlttqcKJFkeCy6
or 
https://github.com/deep-fingerprinting/df


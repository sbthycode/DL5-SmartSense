# DL5-SmartSense
 Solution for Round 3 | Smartsense Test

## Some critical Observations:

1. The judgement and summary files are sorted within the folders. 
2. There is a one to one mapping from the judgement to summary files.
3. In the case of IN-Ext we will be going with the A1 summary.(Very likely A1 and A2 are the summaries produced by separate entities, thus we will be moving ahead with the one produced by A1)

## Exploratory Data Analysis

1. Much of the EDA has been done in the Jupyter Notebook in src directory. It includes code along with the derived plots and conclusions from the same.

Here are some of the self-explanatory graphs(the conclusion of which can be found in the Notebook)
 ![alt text](https://github.com/sbthycode/DL5-SmartSense/blob/3280d7c62e60314c97c26ebabe46baed75871af0/EDA1.png?raw=true)

 ![alt text](https://github.com/sbthycode/DL5-SmartSense/blob/3280d7c62e60314c97c26ebabe46baed75871af0/EDA2.png?raw=true)

  ![alt text](https://github.com/sbthycode/DL5-SmartSense/blob/3280d7c62e60314c97c26ebabe46baed75871af0/WordCloud.png?raw=true)


## Conclusions from the EDA

1. We have (text,summary) pair available from three domains. 
2. There is imbalance in the number of datapoints in each of these domains.
3. Clearly IN-Abs has the most number of datapoints.
4. There is no disparity in the number of texts and their corresponding summaries.


## What model has been used and why?

I have done transfer learning on T5-small. Why small? Because this was a time constrained coding round, where I cannot afford to risk time.

## Results:

## To run the model on your machine:

1. Make sure you have trnsformers and torch installed.
2. Clone the repository.
3. Update the model path in inference.py file
4. Run the inference.py file
5. Give the text in command line input
6. You will see the output of the text in the command line itself.


## How to run for yourself?

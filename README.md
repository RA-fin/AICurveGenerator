# AICurveGenerator
A generative AI model that can generate a CHF risk free curve from a narrative. It can run on a local machine

## Purpose
This tool provides a GenAI model fine tuned to create an interest rate curve from a narrative. It is currently trained on the Nelson-Siegel-Svensson curve of Swiss government bond, but it can be further extended to other curves or output variables.

## Training
The original training dataset is taken from the quarterly economic bullettins of the Swiss National Bank and the corresponding curve in the same month.

The scenario narrative is obtained by downloading the PDF, filtering only the sentences containing an economic summary using a Bert model, summarizing them and finally removing semantic duplicates. The corresponding curve for a given narrative is the one of the 15th of the previous month of the bullettin. The training set is further extended with artificial scenarios generated, taking the original historical scenarios as input, by ChatGPT  for a total scenario count of 2000.

The training can be performed on a decent laptop without even using a GPU.

## Repository content
The repository contains the following folders:

- Data retrieval contains the code to download the narratives from the SNB website and the NSS curves

- Train and infer contains the training code, the code to generate the curves and the pre-trained parameters in the folder nss_qwen25_1_5b_lora

## Usage
The scripts are ready to use, provided that the models are stored somewhere in your system. The examples are contained at the beginning of each script


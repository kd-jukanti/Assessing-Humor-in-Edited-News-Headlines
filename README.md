# Assessing-Humor-in-Edited-News-Headlines
With an idea inspired from a competition task SemEval-2020 “Assessing Humor in Edited News Headlines”, the motivation for this project is to build a model that has the ability to predict the funniness of edited headlines, thereby providing a way to automatically detect humor. This project describes the implementation of three different deep learning models namely, Feed Forward Neural network, Bidirectional Encoder Representations from Transformers (BERT) and Recurrent Neural Network on the humor dataset and compare the results.


In the field of artificial intelligence, humor detection still continues to be a challenging task. Even in the real world, humans sometimes face difficulties in being funny or detecting humor. Humor happens at various intensities where certain jokes are much funnier than others, and therefore developing a system that has the ability to assess the funniness in a given text is more complicated as the system should be able to perceive the relationship across entities and objects. 

## Data 

The text corpus carried out for this work is “Humicroedit” [2], a novel dataset for research in computational humor. The dataset contains news headlines collected from news media posted on Reddit. Each edited headline is scored by five judges where they assign a grade numerically on a scale of 0-3 of funniness scale, where 0 being non-funny and 3 being the most funny. The quality and the ground truth funniness is determined by computing the mean of the five funniness grades. The
resulting dataset contains 15,095 edited headlines with numerically evaluated humor.


The dataset and the edited headlines looks like the following:

<img width="421" alt="Screenshot 2022-03-11 at 12 07 00 PM" src="https://user-images.githubusercontent.com/101395346/157855739-daf21e69-9dfd-4d69-a336-6aa14708ffed.png">

<img width="424" alt="Screenshot 2022-03-11 at 12 06 28 PM" src="https://user-images.githubusercontent.com/101395346/157855744-48afc2a6-008d-47d0-b08f-2086926a9553.png">

## Results 

The results of all the models implemented on the data is shown below. The RMSE score obtained acts as the final metric in determining which model has performed the best. To summarize theresults, the fine-tuned BERT model has proved its efficiency over the other models as it has gained the RMSE score of **0.515**.

<img width="321" alt="Screenshot 2022-03-11 at 12 08 11 PM" src="https://user-images.githubusercontent.com/101395346/157855943-7e2a1d22-9a97-43ae-80e1-39440276dd29.png">


## Conclusion 

Analyzing the funniness in news headlines or in any text and computing the degree of the funniness is a challenging and a critical task. This work is focused on applying different deep learning models and measure the quality of the predictions. The results showed that the fine-tuned BERT model is the better model for our data and study.

# TD Rotman FinHub TDMDAL Hackathon

## Feb. 28, 2020 to Mar. 1, 2020.
> Finalist Group (Top 5)

### Task
In this project, we developed an algorithm extracting information from the transcript of **earning calls** and predicting stcok price movement (**returns**) on the next trading day.

### Methodology
1. We splitted each transcript into the **manager discussion** and **Q&A** parts;
2. For each part, we measured the emotions using Loughran McDonald dictionary;
3. Random forests, support vector regressions, and XGBoost are used to predict returns;
4. Raw predictions on test set are scaled up so that they have the same variance as the training set.

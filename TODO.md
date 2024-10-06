Batch over single text and batches of text
Make everything into a docker container to resolve dependency conflicts with flash attention


Do an ablation test on the "mistake" (giving the model the token we want it to predict in the context and rating it on well it is able to predict that token). Also try setting the perplexity to 1 and just testing the inverse of the cross perplexity as the metric

Make the number of context tokens that the model starts with a hyperparameter because just giving the model 1 word of context is going to skew the perplexity

Fix up the validation pipeline and create a finetuning pipeline to do finetuning scaling laws
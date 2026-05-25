# Tuned halftime performance --- Italy

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | None | 0.61 | 0.54 | 0.56 | 0.56 | C=0.05, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | None | 0.62 | 0.55 | 0.56 | 0.56 | subsample=0.7, reg_lambda=0.5, reg_alpha=0.0, n_estimators=300, min_child_weight=7, max_depth=4, learning_rate=0.03, colsample_bytree=0.7 |
| Random Forest | None | None | 0.62 | 0.53 | 0.57 | 0.56 | n_estimators=600, min_samples_split=12, min_samples_leaf=4, max_features=sqrt, max_depth=12, class_weight=None |
| SVM | Std. | None | 0.60 | 0.46 | 0.52 | 0.52 | max_iter=20000, class_weight=balanced, C=0.3 |
| Naive Bayes | Min-Max | None | 0.51 | 0.52 | 0.58 | 0.53 | var_smoothing=1e-10 |
| FNN | Std. | None | 0.62 | 0.55 | 0.58 | 0.57 | max_iter=600, learning_rate_init=0.003, hidden_layer_sizes=[128, 64, 64], early_stopping=True, batch_size=128, alpha=0.001, activation=tanh |

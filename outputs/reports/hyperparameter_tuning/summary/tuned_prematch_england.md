# Tuned prematch performance --- England

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | None | 0.53 | 0.45 | 0.47 | 0.47 | C=0.05, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | None | 0.54 | 0.45 | 0.46 | 0.47 | subsample=0.7, reg_lambda=0.5, reg_alpha=0.0, n_estimators=300, min_child_weight=7, max_depth=4, learning_rate=0.03, colsample_bytree=0.7 |
| Random Forest | None | None | 0.54 | 0.41 | 0.42 | 0.46 | n_estimators=300, min_samples_split=12, min_samples_leaf=9, max_features=log2, max_depth=12, class_weight=None |
| SVM | Std. | None | 0.52 | 0.37 | 0.43 | 0.42 | max_iter=20000, class_weight=None, C=0.3 |
| Naive Bayes | Min-Max | NM | 0.52 | 0.38 | 0.44 | 0.43 | var_smoothing=1e-06 |
| FNN | Std. | None | 0.54 | 0.43 | 0.52 | 0.46 | max_iter=600, learning_rate_init=0.003, hidden_layer_sizes=[256, 128, 64], early_stopping=True, batch_size=128, alpha=0.5, activation=relu |

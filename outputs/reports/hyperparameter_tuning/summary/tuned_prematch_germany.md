# Tuned prematch performance --- Germany

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | None | 0.43 | 0.39 | 0.44 | 0.42 | C=0.1, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | None | 0.51 | 0.43 | 0.46 | 0.46 | subsample=0.7, reg_lambda=1.5, reg_alpha=0.9, n_estimators=500, min_child_weight=1, max_depth=8, learning_rate=0.03, colsample_bytree=1.0 |
| Random Forest | None | None | 0.53 | 0.40 | 0.38 | 0.45 | n_estimators=300, min_samples_split=12, min_samples_leaf=9, max_features=log2, max_depth=12, class_weight=None |
| SVM | Std. | None | 0.44 | 0.31 | 0.39 | 0.36 | max_iter=20000, class_weight=balanced, C=0.3 |
| Naive Bayes | Min-Max | None | 0.50 | 0.50 | 0.56 | 0.52 | var_smoothing=1e-10 |
| FNN | Std. | SVM-SMOTE | 0.50 | 0.47 | 0.48 | 0.48 | max_iter=600, learning_rate_init=0.0003, hidden_layer_sizes=[128, 64, 64], early_stopping=True, batch_size=64, alpha=0.1, activation=relu |

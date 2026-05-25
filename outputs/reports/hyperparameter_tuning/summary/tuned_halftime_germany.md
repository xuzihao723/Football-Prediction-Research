# Tuned halftime performance --- Germany

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | NM | 0.55 | 0.54 | 0.57 | 0.55 | C=0.05, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | None | 0.64 | 0.59 | 0.61 | 0.60 | subsample=0.85, reg_lambda=1.0, reg_alpha=0.2, n_estimators=300, min_child_weight=1, max_depth=10, learning_rate=0.03, colsample_bytree=0.7 |
| Random Forest | None | None | 0.64 | 0.59 | 0.60 | 0.60 | n_estimators=300, min_samples_split=2, min_samples_leaf=4, max_features=sqrt, max_depth=24, class_weight=balanced |
| SVM | Std. | SMOTE-NN | 0.59 | 0.54 | 0.56 | 0.57 | max_iter=20000, class_weight=None, C=1.0 |
| Naive Bayes | Min-Max | None | 0.52 | 0.52 | 0.59 | 0.54 | var_smoothing=1e-10 |
| FNN | Std. | NM | 0.61 | 0.56 | 0.57 | 0.58 | max_iter=600, learning_rate_init=0.0006, hidden_layer_sizes=[128, 64, 64], early_stopping=True, batch_size=64, alpha=0.5, activation=tanh |

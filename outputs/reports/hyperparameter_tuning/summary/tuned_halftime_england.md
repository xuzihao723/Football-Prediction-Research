# Tuned halftime performance --- England

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | None | 0.64 | 0.57 | 0.59 | 0.58 | C=0.1, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | None | 0.65 | 0.58 | 0.60 | 0.59 | subsample=0.85, reg_lambda=0.5, reg_alpha=0.0, n_estimators=300, min_child_weight=7, max_depth=6, learning_rate=0.03, colsample_bytree=1.0 |
| Random Forest | None | ROS | 0.66 | 0.60 | 0.60 | 0.61 | n_estimators=1000, min_samples_split=2, min_samples_leaf=4, max_features=sqrt, max_depth=46, class_weight=None |
| SVM | Std. | ROS | 0.64 | 0.54 | 0.57 | 0.57 | max_iter=20000, class_weight=None, C=0.7 |
| Naive Bayes | Min-Max | None | 0.56 | 0.53 | 0.54 | 0.53 | var_smoothing=1e-10 |
| FNN | Std. | None | 0.64 | 0.57 | 0.59 | 0.58 | max_iter=600, learning_rate_init=0.0003, hidden_layer_sizes=[64], early_stopping=True, batch_size=128, alpha=0.01, activation=tanh |

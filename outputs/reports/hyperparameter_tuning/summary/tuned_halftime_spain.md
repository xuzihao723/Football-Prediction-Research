# Tuned halftime performance --- Spain

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | None | 0.59 | 0.54 | 0.54 | 0.55 | C=0.05, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | SMOTE-Tomek | 0.60 | 0.56 | 0.56 | 0.57 | subsample=0.85, reg_lambda=1.0, reg_alpha=0.9, n_estimators=300, min_child_weight=1, max_depth=8, learning_rate=0.03, colsample_bytree=0.85 |
| Random Forest | None | None | 0.61 | 0.54 | 0.56 | 0.56 | n_estimators=600, min_samples_split=12, min_samples_leaf=4, max_features=sqrt, max_depth=12, class_weight=None |
| SVM | Std. | ROS | 0.59 | 0.50 | 0.52 | 0.53 | max_iter=20000, class_weight=None, C=0.3 |
| Naive Bayes | Min-Max | SMOTE-Tomek | 0.56 | 0.49 | 0.52 | 0.51 | var_smoothing=1e-10 |
| FNN | Std. | None | 0.58 | 0.52 | 0.53 | 0.54 | max_iter=600, learning_rate_init=0.0003, hidden_layer_sizes=[64], early_stopping=True, batch_size=128, alpha=0.01, activation=tanh |

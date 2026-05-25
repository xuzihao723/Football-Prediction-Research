# Tuned prematch performance --- Spain

| Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Std. | None | 0.50 | 0.43 | 0.45 | 0.45 | C=0.05, class_weight=None, max_iter=3000, penalty=l2, solver=lbfgs |
| XGBoost | None | None | 0.52 | 0.43 | 0.46 | 0.45 | subsample=0.85, reg_lambda=1.0, reg_alpha=0.2, n_estimators=300, min_child_weight=1, max_depth=10, learning_rate=0.03, colsample_bytree=0.7 |
| Random Forest | None | None | 0.54 | 0.44 | 0.48 | 0.47 | n_estimators=600, min_samples_split=12, min_samples_leaf=4, max_features=sqrt, max_depth=12, class_weight=None |
| SVM | Std. | None | 0.50 | 0.35 | 0.41 | 0.41 | max_iter=20000, class_weight=balanced, C=0.3 |
| Naive Bayes | Min-Max | ROS | 0.52 | 0.41 | 0.46 | 0.45 | var_smoothing=1e-10 |
| FNN | Std. | None | 0.51 | 0.39 | 0.41 | 0.44 | max_iter=600, learning_rate_init=0.003, hidden_layer_sizes=[256, 128, 64], early_stopping=True, batch_size=128, alpha=0.5, activation=relu |

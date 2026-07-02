# Programmatic API Specs

### `src.model.train_pipeline(data_path: str) -> None`
Loads dataset, executes preprocessing scaling, trains classifier, and saves calibrated model.

### `src.explain.get_shap_values(model, X) -> tuple`
Computes local and global SHAP metrics for feature inputs.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)
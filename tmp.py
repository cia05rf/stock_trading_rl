from training.inference import Infer
infer = Infer(model_path="training/models/ppo_stock_trading_20251214_1636.zip")
preds, _ = infer.infer_date("2024-01-15")
print("Action counts:\n", preds["action_type"].value_counts())
print("\nTop buys:\n", preds[preds.action_type=="buy"].head())
print("\nTop sells:\n", preds[preds.action_type=="sell"].head())
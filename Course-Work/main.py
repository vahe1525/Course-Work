import pandas as pd
import numpy as np
import random
import math
from Random_Forest_algorithm import RandomForest, Evaluate
from Visualize import plot_feature_importance, plot_confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("reduced_data.csv")
#df = pd.read_csv(r"C:\Users\VaheKarmirshalyan\source\repos\vahe1525\Course-Work\Dataset analyze kursayin\reduced_data.csv")
target_col = "diagnosis"

# #splitting data into train and test sets using sklearn's train_test_split for better stratification
# X = df.drop(columns=[target_col]) 
# y = df[target_col]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# # Քայլ 3 — Վերադարձնում ենք քո ալգորիթմի համար list-of-dicts ֆորմատ
# train_data = pd.concat([X_train, y_train], axis=1).to_dict(orient="records")
# test_data = pd.concat([X_test, y_test], axis=1).to_dict(orient="records")

# Փոխակերպում մեր ալգորիթմի համար հարմար ձևաչափի
dataset = df.to_dict(orient="records")

random.seed(42)
random.shuffle(dataset)
split_index = int(0.8 * len(dataset))
    
train_data = dataset[:split_index]
test_data = dataset[split_index:]

# Որոշում ենք n_features-ը՝ sqrt(ընդհանուր հատկանիշներ) դասական կանոնով
total_features = len(df.columns) - 1 # -1-ը 'diagnosis'-ի համար
n_features_sqrt = int(math.sqrt(total_features))

#Creating Model
rf_model = RandomForest(n_estimators = 50, max_depth = 5 , min_samples_split = 5, criterion = "gini", n_features = n_features_sqrt)
# rf_model = RandomForest(n_estimators = 50, max_depth = 10, min_samples_split = 5, criterion = "entropy", n_features = n_features_sqrt)

rf_model.fit(train_data, target_col)

predictions = rf_model.predict(test_data)

# Օգտագործում ենք մեր իսկ գրած գնահատման ֆունկցիան
metrics = Evaluate(test_data, predictions, target_col)

print("\n--- 🏁 ԱՎԱՐՏՎԱԾ Է։ ԳՆԱՀԱՏՄԱՆ ԱՐԴՅՈՒՆՔՆԵՐ (ՔԱՅԼ 4) ---")
print(f"  Accuracy:  {metrics['Accuracy']}")
print(f"  Precision: {metrics['Precision']}")
print(f"  Recall:    {metrics['Recall']}")
print(f"  F1-score:  {metrics['F1-score']}")
print(f"  Confusion Matrix: {metrics['Confusion_Matrix']}")
print("-------------------------------------------------")

print("\n--- 📊 ՔԱՅԼ 5. CONFUSION MATRIX ---")
plot_confusion_matrix(metrics)

print("\n--- 📈 ՔԱՅԼ 7. ՀԱՏԿԱՆԻՇՆԵՐԻ ԿԱՐԵՎՈՐՈՒԹՅԱՆ ԳԾԱՊԱՏԿԵՐ ---")
plot_feature_importance(rf_model.feature_importances_, top_n=10, title="Random Forest: Լավագույն 10 ազդեցիկ հատկանիշները")

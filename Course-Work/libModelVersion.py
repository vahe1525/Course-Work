from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import pandas as pd
import math

def run_sklearn_benchmark(df, target_col):
    
    print("\n--- 🔬 ԲԵՆՉՄԱՐՔ. SCiKIT-LEARN RANDOM FOREST ---")
    
    # 1. Տվյալների Բաժանում (X և y)
    # Ձեր մոդելը աշխատում էր list[dict]-ով, sklearn-ը՝ DataFrame-ով
    X = df.drop(columns=[target_col]) # Հատկանիշներ (բոլոր սյունակները, բացի թիրախից)
    y = df[target_col] # Թիրախային սյունակ

    # 2. Train/Test Բաժանում
    # Օգտագործում ենք sklearn-ի ֆունկցիան, որը ավելի պրոֆեսիոնալ է և stratify է անում
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Մոդելի Ստեղծում
    # Օգտագործում ենք նույն պարամետրերը, ինչ մեր մոդելինը
    # n_features-ի փոխարեն sklearn-ը պահանջում է max_features (որը մենք տալիս ենք քանակով)
    total_features = X_train.shape[1]
    n_features_sqrt = int(math.sqrt(total_features)) # Օգտագործում ենք նույն sqrt կանոնը

    sk_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10, 
        min_samples_split=5, 
        criterion="gini", 
        max_features=n_features_sqrt, # Sklearn-ի անվանումը 'n_features'-ի համար
        random_state=42 # Կարևոր է համեմատության համար
    )

    # 4. Ուսուցում
    start_time = time.time()
    sk_model.fit(X_train, y_train)
    end_time = time.time()
    
    # 5. Կանխատեսում և Գնահատում
    y_pred = sk_model.predict(X_test)
    
    # Գնահատում sklearn-ի ֆունկցիաներով
    sk_accuracy = accuracy_score(y_test, y_pred)
    sk_precision = precision_score(y_test, y_pred)
    sk_recall = recall_score(y_test, y_pred)
    sk_f1 = f1_score(y_test, y_pred)
    sk_cm = confusion_matrix(y_test, y_pred)

    # 6. Արդյունքների Ցուցադրում
    print(f"  > Օգտագործված ծառեր: {sk_model.n_estimators}")
    print(f"  > Օգտագործված հատկանիշներ (max_features): {sk_model.max_features}")
    print(f"  > Ուսուցման ժամանակը: {end_time - start_time:.2f} վայրկյան")
    print(f"\n  Accuracy (Sklearn): {sk_accuracy:.3f}")
    print(f"  Precision (Sklearn): {sk_precision:.3f}")
    print(f"  F1-score (Sklearn): {sk_f1:.3f}")
    print(f"  Recall (Sklearn): {sk_recall:.3f}")
    print(f"\n  Confusion Matrix (Sklearn):\n {sk_cm}")
    

    
#df = pd.read_csv(r"C:\Users\VaheKarmirshalyan\source\repos\vahe1525\Course-Work\Dataset analyze kursayin\reduced_data.csv")
df = pd.read_csv("reduced_data.csv")
target_col = "diagnosis"

run_sklearn_benchmark(df, target_col)


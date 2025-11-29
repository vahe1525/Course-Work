import matplotlib.pyplot as plt
import numpy as np
import re


def plot_feature_importance(importances, top_n=10, title="Հատկանիշների Կարևորությունը"):
    """
    Կառուցում է Bar Chart՝ ցույց տալով Feature Importance-ի միավորները։
    
    importances: Այն տեսակավորված ցուցակն է, որը ստացանք rf_model.feature_importances_-ից։
    """
    if not importances:
        print("Վիզուալացման տվյալներ չկան։")
        return

    # Վերցնում ենք միայն լավագույն N հատկանիշները
    top_importances = importances[:top_n]
    
    # Տարանջատում ենք անուններն ու միավորները
    features = [item[0] for item in top_importances]
    scores = [item[1] for item in top_importances]
    
    # 1. Գծապատկերի ստեղծում
    plt.figure(figsize=(12, 6))
    
    # 2. Սյունակների կառուցում
    plt.barh(features, scores, color='#2c7bb6')
    
    # 3. Ձևավորում
    plt.xlabel("Կարևորության Միավոր (Միջին Information Gain)")
    plt.title(title)
    plt.gca().invert_yaxis() # Լավագույնը վերևում ցուցադրելու համար
    
    # Ավելացնում ենք թվերը սյունակների վրա
    for index, value in enumerate(scores):
        plt.text(value, index, f" {value:.4f}")
        
    plt.show()


def plot_confusion_matrix(metrics_dict, target_names=['Առողջ (0)', 'Հիվանդ (1)']):
    """
    Կառուցում է Confusion Matrix-ի տեսողական պատկերը։
    """
    
    # 1. Թվերի դուրսբերում (Parsing)
    # Մենք գիտենք, որ տողն ունի այս տեսքը՝ "TP:45, FP:1, FN:6, TN:62"
    matrix_str = metrics_dict['Confusion_Matrix']
    
    # Օգտագործում ենք regex (կամ պարզ split), որպեսզի ստանանք թվերը
    # Ավելի պարզ է՝ բերենք տվյալները կանոնավոր տեսքի
    try:
        numbers = [int(n) for n in re.findall(r'\d+', matrix_str)]
        TP, FP, FN, TN = numbers[0], numbers[1], numbers[2], numbers[3]
    except (IndexError, ValueError):
        print("Վիզուալացման տվյալների անհամապատասխանություն։")
        return

    # 2. Մատրիցի ստեղծում
    cm = np.array([[TN, FP], [FN, TP]])
    
    # 3. Պատկերների ստեղծում
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # Կապույտ գունապնակ
    plt.title('Confusion Matrix', fontsize=15)
    plt.colorbar()
    
    # 4. Առանցքների դասավորում
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    # 5. Թվերի ցուցադրում
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Թիվը սպիտակ կամ սև է՝ կախված ֆոնի գույնից
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=18)

    plt.ylabel('Իրական Դաս (True Label)')
    plt.xlabel('Կանխատեսված Դաս (Predicted Label)')
    plt.tight_layout()
    plt.show()
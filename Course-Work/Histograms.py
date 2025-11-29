import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv")

# Գտնում ենք բոլոր mean հատկանիշները
mean_features = [col for col in df.columns if '_mean' in col]

# Ստեղծում ենք գրաֆիկներ ամեն հատկանիշի համար
for feature in mean_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=feature, hue="diagnosis", kde=True, stat="density", common_norm=False)
    plt.title(f"{feature.replace('_', ' ').title()} Distribution by Diagnosis")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend(title="Diagnosis", labels=["Malignant (M)", "Benign (B)"])
    plt.show()


    # շատ լավ այս բաշխումները ինչ որ չափով հասկացա արի անցնենք հաջորդ քայլերին հետագայում եթե կարիքը լինի հետ կգանք և նաև օգնես այս ամենի մասին գրենք փաստաթղթում
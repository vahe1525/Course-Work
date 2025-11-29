import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# df = pd.read_csv("data.csv")

# print(df.shape)

# print(df.head())

# # Տեղեկություն սյուների մասին
# print(df.info())

# # Բացակայող արժեքների ստուգում
# print(df.isnull().sum())
# # Թիրախային փոփոխականի բաշխում
# print("Diagnosis distribution (count):")
# print(df['diagnosis'].value_counts())

# print("\nDiagnosis distribution (percentage):")
# print(df['diagnosis'].value_counts(normalize=True) * 100)

# # Անպիտան սյունը հեռացնել
# df = df.drop(columns=['id'], errors='ignore')

# print(df.head())
# le = LabelEncoder()
# df['diagnosis'] = le.fit_transform(df['diagnosis'])
# print(df.head())
# df.to_csv("data_cleaned.csv", index=False)



# #bashxumneri histogramner

# sns.countplot(x='diagnosis', data=df)
# plt.title('Բարորակ և չարորակ դեպքերի բաշխում')
# plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
# plt.ylabel('Քանակ')


# df.drop('diagnosis', axis=1).hist(figsize=(15,15), bins=30)
# plt.suptitle('Հատկանիշների բաշխումներ')

# # ավելացնում ենք տարածությունը subplot-ների միջև
# plt.subplots_adjust(hspace=0.8, wspace=0.8)

# plt.show()



dfc = pd.read_csv("data_cleaned.csv")

# Կոռելացիոն մատրից
corr_matrix = dfc.corr()

# # Տպում
# print(corr_matrix)

# # Տեսանելիություն heatmap-ի միջոցով
# plt.figure(figsize=(15, 12))
# sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
# plt.title('Հատկանիշների կոռելացիոն մատրից')
# plt.show()


corr_with_target = corr_matrix['diagnosis'].sort_values(ascending=False)
print(corr_with_target)


#cleaning dataset after finding correlation matrix

df = pd.read_csv("data_cleaned.csv")

# Ստանում ենք կոռելացիոն մատրիցան
corr_matrix = df.corr().abs()

# Ստեղծում ենք վերին եռանկյունաձև մատրիցա՝ կրկնությունները չդիտարկելու համար
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Գտնում ենք այն հատկանիշները, որոնց կոռելացիան գերազանցում է 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print("Բարձր կոռելացիայով հատկանիշներ՝")
print(to_drop)

# Ջնջում ենք բարձր կոռելացիայով հատկանիշները
df_reduced = df.drop(columns=to_drop)
print("Նոր տվյալների չափս՝", df_reduced.shape)

# Պահպանում ենք նոր տվյալների հավաքածուն ֆայլում
df_reduced.to_csv("reduced_data.csv", index=False, encoding='utf-8-sig')

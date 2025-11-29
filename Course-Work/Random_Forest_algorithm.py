from ensurepip import bootstrap
import math
import pandas as pd
import random
from collections import Counter
import time

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, result=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.result = result

# Decison tree class
class MyDecisionTree:
    
    def __init__(self, max_depth = 10, min_samples_split = 2, criterion = "gini", n_features = None):
        self.root = None
        self.target_col = None

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.n_features = n_features
        self.feature_importances = {} 

    def _calculate_entropy(self, rows):
        total = len(rows)
        if total == 0:
            return 0
        counts = {}
        for row in rows:
            label = row[self.target_col]
            counts[label] = counts.get(label, 0) + 1
        
        impurity = 0
        for label in counts:
            prob = counts[label] / total
            if prob > 0:
                impurity -= prob * math.log2(prob)
        return impurity

    def _calculate_gini(self, rows):
            total = len(rows)
            if total == 0:
                return 0
            counts = {}
            for row in rows:
                label = row[self.target_col] 
                counts[label] = counts.get(label, 0) + 1
            impurity = 1
            for label in counts:
                prob = counts[label] / total
                impurity -= prob ** 2
            return impurity

    def _split_dataset(self, rows, feature, threshold):
        left = [row for row in rows if row.get(feature) is not None and row[feature] < threshold]
        right = [row for row in rows if row.get(feature) is not None and row[feature] >= threshold]
        return left, right 

    def _best_split(self, rows):
        best_gain = 0
        best_feature = None
        best_threshold = None 

        if self.criterion == "gini":
            current_impurity = self._calculate_gini(rows)
        else:
            current_impurity = self._calculate_entropy(rows)

        if not rows:
            return 0, None, None
            
        all_features = [col for col in rows[0] if col != self.target_col]
        
        if self.n_features is None:
            features_to_check = all_features
        else:
            num_to_sample = min(self.n_features, len(all_features))
            features_to_check = random.sample(all_features, num_to_sample)

        for feature in features_to_check:
            unique_values = sorted(set([row[feature] for row in rows if row.get(feature) is not None]))
            
            threshold_candidates = []
            for i in range(len(unique_values) - 1):
                midpoint = (unique_values[i] + unique_values[i+1]) / 2
                threshold_candidates.append(midpoint)
            
            for threshold in threshold_candidates:
                left, right = self._split_dataset(rows, feature, threshold)
                if len(left) == 0 or len(right) == 0:
                    continue
                
                p = len(left) / len(rows)
                if self.criterion == "gini":
                    gain = current_impurity - p * self._calculate_gini(left) - (1 - p) * self._calculate_gini(right)
                else:
                    gain = current_impurity - p * self._calculate_entropy(left) - (1 - p) * self._calculate_entropy(right)

                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_gain, best_feature, best_threshold

    def fit(self, rows, target_col):
            # 1. Պահպանում ենք թիրախի անունը, որ մյուս մեթոդներն օգտագործեն
            self.target_col = target_col
        
            # 2. Գործարկում ենք ռեկուրսիվ ուսուցումը և արդյունքը պահում self.root-ում
            self.root = self._build_tree(rows,depth = 0)

    def _build_tree(self, rows, depth):
            gain, feature, threshold = self._best_split(rows)

            if gain > 0 and feature is not None:
                # Գումարում ենք Gain-ը տվյալ հատկանիշների ընդհանուր կարևորության համար
                self.feature_importances[feature] = self.feature_importances.get(feature, 0) + gain

            if (gain == 0) or (depth >= self.max_depth) or (len(rows) < self.min_samples_split) :
                labels = [row[self.target_col] for row in rows]
                return DecisionNode(result=max(set(labels), key=labels.count))

            left, right = self._split_dataset(rows, feature, threshold)
            left_branch = self._build_tree(left, depth = depth + 1)
            right_branch = self._build_tree(right, depth = depth + 1)
            return DecisionNode(feature, threshold, left_branch, right_branch)

    def predict(self, rows):
            predictions = []
            for row in rows:
                pred = self._predict_row(row, self.root)
                predictions.append(pred)
            return predictions

    def _predict_row(self, row, node):
        if node.result is not None:
            return node.result
            
        if row.get(node.feature) is not None and row[node.feature] < node.threshold:
            branch = node.left
        else:
            branch = node.right
        
        return self._predict_row(row, branch) 
    

class RandomForest:
    def __init__(self, n_estimators = 100, max_depth=10, min_samples_split=2, criterion="gini", n_features=None):
        self.n_estimators = n_estimators
        self.trees = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.n_features = n_features
        self.training_time = 0.0
        
        self.feature_importances = None

        self.trees = []

    def fit(self, rows, target_col):
        self.trees = []
        self.target_col = target_col

        n_samples = len(rows)

        total_importances = {}

        print(f"Սկսվում է {self.n_estimators} ծառերից բաղկացած անտառի ուսուցումը...")
        start_time = time.time()

        for i in range(self.n_estimators):
            bootstrap_sample = []

            for _ in range(n_samples):
                random_row = random.choice(rows)
                bootstrap_sample.append(random_row)

            tree = MyDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion=self.criterion, n_features=self.n_features)

            tree.fit(bootstrap_sample, self.target_col)

            for feature, score in tree.feature_importances.items():
                total_importances[feature] = total_importances.get(feature, 0) + score
            
            self.trees.append(tree)
            
            print(f"   - Ծառ {i+1}/{self.n_estimators} ստեղծված է։")

        end_time = time.time()
        self.training_time = end_time - start_time
        print(f"✅ Անտառի ուսուցումն ավարտված է։ train time : {self.training_time:.2f} seconds ")

        # 1. Միջինացնում ենք՝ բաժանելով ծառերի քանակի վրա
        average_importances = {
            k: v / self.n_estimators for k, v in total_importances.items()
        }
        
        # 2. Փոխակերպում ենք այն տեսակավորված ցուցակի (հեշտ գծելու համար)
        sorted_importances = sorted(
            average_importances.items(), key=lambda item: item[1], reverse=True
        )
        
        # 3. Պահպանում ենք վերջնական արդյունքը
        self.feature_importances_ = sorted_importances

    def predict(self, rows):
        final_predictions_for_all_rows = []
        
        for row in rows:
            
            predictions_for_this_row = []
            
            for tree in self.trees:
                prediction = tree.predict([row])[0]
                
                predictions_for_this_row.append(prediction)
            
            
            vote_counts = Counter(predictions_for_this_row)
            final_prediction = vote_counts.most_common(1)[0][0]
            
            final_predictions_for_all_rows.append(final_prediction)
            
        return final_predictions_for_all_rows


def Evaluate(test_rows, predictions, target_col) :
    tp = tn = fp = fn = 0
    for i, row in enumerate(test_rows):
        y_true = row[target_col]
        y_pred = predictions[i]
        
        if y_true == 1 and y_pred == 1: tp += 1
        elif y_true == 0 and y_pred == 0: tn += 1
        elif y_true == 0 and y_pred == 1: fp += 1
        elif y_true == 1 and y_pred == 0: fn += 1

    accuracy = (tp + tn) / len(test_rows) if len(test_rows) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-score": round(f1, 3),
        "Confusion_Matrix": f"TP:{tp}, FP:{fp}, FN:{fn}, TN:{tn}"
    }
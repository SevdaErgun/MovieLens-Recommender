from surprise import Dataset
from surprise.model_selection import KFold
from surprise import KNNWithMeans
from surprise import accuracy
import math
from collections import defaultdict

#Helper Class for Step 6
class NDCGEvaluator:
    def __init__(self, k=10):
        self.k = k
    
    def _dcg(self, relevances):
        dcg = 0.0
        for rank, rel in enumerate(relevances, start=1):
            dcg += rel / math.log2(rank + 1)
        return dcg

    def calculate_ndcg(self, predictions):
        user_item_data = defaultdict(list)

        # Group predictions by user
        for uid, iid, true_r, est, _ in predictions:
            user_item_data[uid].append((est, true_r))
        
        ndcg_values = []

        for uid, items in user_item_data.items():

            # Sort items by estimated rating in descending order
            ranked_items = sorted(items, key=lambda x: x[0], reverse=True)

            # Take top-k items based on estimated ratings
            top_k_relevance = [true_r for (_, true_r) in ranked_items[:self.k]]

            # Calculate DCG
            dcg = self._dcg(top_k_relevance)

            # All true ratings for the user
            all_true_ratings = [true_r for (_, true_r) in items]
            
            # Sort all true ratings to get ideal ranking
            ideal_relevance = sorted(all_true_ratings, reverse=True)[:self.k]

            # Calculate IDCG
            idcg = self._dcg(ideal_relevance)

            # Calculate NDCG
            if idcg == 0:
                ndcg = 0.0
            else:
                ndcg = dcg / idcg
            
            ndcg_values.append(ndcg)

        # Average NDCG across all users
        return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

#Helper Method for Step 2
def produce_predictions_to_test_items(data, knn, cv):
    
    return predictions

#Helper Method for Step 4
def get_top_n_filtered(predictions, n=10, threshold=3.5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

#Helper Method for Step 5
def calculate_precision_recall(top_n_list, testset, threshold=3.5):
    user_true_ratings = defaultdict(list)
    for uid, iid, true_r in testset:
        user_true_ratings[uid].append((iid, true_r))

    precisions = dict()
    recalls = dict()
    
    all_test_users = user_true_ratings.keys()

    for uid in all_test_users:
        user_recs = top_n_list.get(uid, [])
        recommended_iids = [iid for (iid, _) in user_recs]

        user_true = user_true_ratings[uid]
        relevant_iids = [iid for (iid, r) in user_true if r >= threshold]

        n_rel_and_rec = len(set(recommended_iids).intersection(set(relevant_iids)))
        n_rec = len(recommended_iids)
        n_rel = len(relevant_iids)

        precisions[uid] = n_rel_and_rec / n_rec if n_rec > 0 else 0
        recalls[uid] = n_rel_and_rec / n_rel if n_rel > 0 else 0

    avg_p = sum(precisions.values()) / len(precisions) if precisions else 0
    avg_r = sum(recalls.values()) / len(recalls) if recalls else 0
    return avg_p, avg_r

#Helper Method For Printing Top-N Recommendations
def print_top_n_recs(top_n, user_id, n=10):
    if user_id not in top_n:
        print(f"For the User {user_id} Top-{n} could not found.")
        return

    print(f"\n=== TOP-{n} RECOMMENDATIONS FOR USER {user_id} ===")
    for rank, (iid, est) in enumerate(top_n[user_id][:n], 1):
        print(f"{rank:>2}. Movie {iid} | Rating: {est:.3f}")
    print()


#Step 1 : Load Dataset and K-Fold Cross Validation
data = Dataset.load_builtin('ml-100k')
cv = KFold(n_splits=5, random_state=42, shuffle=True)

#--------------------------------------------------------------------------------

#Step 2 : User-Based K-Nearest Neighbor Collaborative Filtering Approach 
sim_options =  {'name': 'pearson', 'user_based':True}
knn = KNNWithMeans(k=50, sim_options=sim_options, verbose=False)

counter = 1
for train, test in cv.split(data):
    print(f"Fold {counter}")

    knn.fit(train)

    predictions = knn.test(test)

    print(f"{len(predictions)} movie predicted.")
    print(f"Predictions : {predictions[:2][:2]}") # İlk 2 tahmini göster

    #--------------------------------------------------------------------------------

    #Step 3 : MAE Calculation
    mae = accuracy.mae(predictions, verbose=False)
    print(f"MAE (Mean Absolute Error): {mae:.4f}")

    #--------------------------------------------------------------------------------

    #Step 4 : Top-10 Recommendation Lists Creation (with Threshold Filtering)
    top_10_list = get_top_n_filtered(predictions, n=10, threshold=3.5)
    print("Top-10 Recommendation Lists (Threshold=3.5)")
    print_top_n_recs(top_10_list, user_id=list(top_10_list.keys())[0], n=10)

    #--------------------------------------------------------------------------------

    #Step 5 : Precision and Recall Calculation for Top-10 Lists
    precison,recall= calculate_precision_recall(top_10_list, test, threshold=3.5)
    print(f"Precision at Top-10: {precison}")
    print(f"Recall at Top-10: {recall}")

    #--------------------------------------------------------------------------------

    #Step 6 : NDCG Calculation using Custom Class
    ndcg = NDCGEvaluator(k=10)
    ndcg_result = ndcg.calculate_ndcg(predictions)
    print(f"NDCG at Top-10: {ndcg_result}")

    counter += 1

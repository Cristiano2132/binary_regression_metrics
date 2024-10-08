from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
def evaluate_ks_and_roc_auc(y_real, y_proba):
    # Unite both visions to be able to filter
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba[:, 1]
    
    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]
    
    ks = ks_2samp(class0['proba'], class1['proba'])
    roc_auc = roc_auc_score(df['real'] , df['proba'])
    
    print(f"KS: {ks.statistic:.4f} (p-value: {ks.pvalue:.3e})")
    print(f"ROC AUC: {roc_auc:.4f}")
    return ks.statistic, roc_auc
print("Good classifier:")
ks_good, auc_good = evaluate_ks_and_roc_auc(y_good, y_proba_good)
print("Medium classifier:")
ks_medium, auc_medium = evaluate_ks_and_roc_auc(y_medium, y_proba_medium)
print("Bad classifier:")
ks_bad, auc_bad = evaluate_ks_and_roc_auc(y_bad, y_proba_bad)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def FeatureImportance(model, features):
  importance = model.feature_importances_
  for i,v in enumerate(importance):
	  print('Feature: %s, Score: %.5f' % (features[i], v))
       
def AccuracyScore(y_test, y_pred):
  print("Accuracy of the model:", accuracy_score(y_test, y_pred))

def ROCCurve(y_test, y_proba):
  fpr, tpr, thresholds = roc_curve(y_test, y_proba)
  plt.plot(fpr, tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
import sys
from datamodel import DataModel

from sklearn import linear_model, svm, cross_validation
from sklearn.feature_extraction import DictVectorizer


def kfold_cross_val(classifier, x, y, cv=5, do_shuffle=True):
  scores, sensitivities, specificities, jaccard_poses, jaccard_negs = [], [], [], [], []
  kf = cross_validation.KFold(len(x), n_folds=cv, shuffle=do_shuffle)
  for train_i, test_i in kf:
    train_x, train_y, test_x, test_y = [], [], [], []
    for index in train_i:
      train_x.append(x[index])
      train_y.append(y[index])
    for index in test_i:
      test_x.append(x[index])
      test_y.append(y[index])
    temp_classifier = classifier
    temp_classifier.fit(train_x, train_y)
    scores.append(temp_classifier.score(test_x, test_y))
    test_y_pred = temp_classifier.predict(test_x)
    true_pos, true_neg, false_pos, false_neg = 0., 0., 0., 0.
    for i in range(len(test_x)):
      if(test_y[i] == 1 and test_y_pred[i] == 1):
        true_pos += 1
      elif(test_y[i] == -1 and test_y_pred[i] == -1):
        true_neg += 1
      elif(test_y[i] == -1 and test_y_pred[i] == 1):
        false_pos += 1
      elif(test_y[i] == 1 and test_y_pred[i] == -1):
        false_neg += 1
    sensitivity = 0.
    if (true_pos+false_neg > 0):
      sensitivity = true_pos/(true_pos+false_neg)
    sensitivities.append(sensitivity)
    specificity = 0.
    if (true_neg+false_pos > 0):
      specificity = true_neg/(true_neg+false_pos)
    specificities.append(specificity)
    jaccard_pos, jaccard_neg = 0., 0.
    if(true_pos+false_pos+false_neg > 0):
      jaccard_pos = true_pos/(true_pos + false_pos + false_neg)
    jaccard_poses.append(jaccard_pos)
    if(true_neg+false_pos+false_neg > 0):
      jaccard_neg = true_neg/(true_neg + false_pos + false_neg)
    jaccard_negs.append(jaccard_neg)
  return {'accuracy':scores, 'sensitivity':sensitivities, \
          'specificity':specificities, 'jaccard_pos':jaccard_poses, \
          'jaccard_neg':jaccard_negs}


def get_rich_featured_training(dm,training):
  vec = DictVectorizer()
  new_training = vec.fit_transform(training).toarray()
  return (new_training, vec.get_feature_names())


def main(argv):
  train_count = -1
  if(len(argv)>0):
    train_count = int(argv[0])
  dm = DataModel()
  dm.get_data(train_count)
#  (training, feature_names) = get_rich_featured_training(dm,lines)
  print(len(dm.data.keys()))
  print(len(dm.train))
  print(len(dm.test))


if  __name__ == '__main__':
  main(sys.argv[1:])


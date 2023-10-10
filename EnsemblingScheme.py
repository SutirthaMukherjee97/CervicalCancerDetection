def choquet_inferencing(X,measure,axis=0):
  sum=X[0]*measure[0]
  for i in range(1,measure.shape[axis]):
    sum+=(measure[i]-measure[i-1])*X[i];
  return sum


def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction


def ensemble_choquet(y_test_1d,pred_rgb,pred_hsv,pred_yuv):
    num_classes = pred_rgb.shape[1]
    Y = np.zeros(pred_rgb.shape,dtype=float)
    for samples in range(pred_rgb.shape[0]):
        for classes in range(pred_rgb.shape[1]):
            X = np.array([pred_rgb[samples][classes], pred_hsv[samples][classes], pred_yuv[samples][classes] ])
            X=np.sort(X)
            # measure = np.array([0.70,0.20,0.10])
            measure=np.array([0.15,0.25,0.60])
            X_agg = choquet_inferencing(X,measure) #check this line later

            Y[samples][classes] = X_agg

    sugeno_pred = predicting(Y)

    correct = np.where(sugeno_pred == y_test_1d)[0].shape[0]
    total = y_test_1d.shape[0]

    print("Accuracy = ",correct/total)
    # classes =  [ "1" , "2" , "3" , "4", "5", "6", "7", "8", "9", "10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32"]
    classes_2=np.arange(0,5)
    classes_2=[str(i) for i in classes_2]
    metrics(y_test_1d,sugeno_pred,classes_2)

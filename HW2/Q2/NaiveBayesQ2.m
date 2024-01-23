%匯入資料
data = load("IMU_label_data.mat");

%資料前處理
X_train = data.IMU_label_data(1:100,1:3);
y_train = data.IMU_label_data(1:100,4);
X_test  = data.IMU_label_data(101:200,1:3);
y_test  = data.IMU_label_data(101:200,4);

%模型訓練
disp(['-----Start  Training-----'])
model = fitcnb(X_train,y_train);
disp(['-----Finish Training-----'])

%模型預測
disp(['-----Start  Predict-----'])
prediction = predict(model,X_test);
disp(['-----Finish Predict-----'])

%模型結果
acc = sum(prediction==y_test)/length(y_test);
fprintf('Accuracy: %f\n',acc);

confusion_table = confusionmat(y_test, prediction);
disp('Confusion Table for Naive Bayes:');
disp('          Predicted 0   Predicted 1');
disp(['Actual 0  ' num2str(confusion_table(1,1)) '             ' num2str(confusion_table(1,2))]);
disp(['Actual 1  ' num2str(confusion_table(2,1)) '             ' num2str(confusion_table(2,2))]);

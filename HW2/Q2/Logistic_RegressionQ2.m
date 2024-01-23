%匯入資料
data = load("IMU_label_data.mat");

%資料前處理
X_train = data.IMU_label_data(1:100,1:3);
y_train = data.IMU_label_data(1:100,4);
X_test  = data.IMU_label_data(101:200,1:3);
y_test  = data.IMU_label_data(101:200,4);

%初始化模型參數及訓練參數
X_train = [ones(size(X_train, 1), 1) X_train]; % 加入Bias
X_test = [ones(size(X_test, 1), 1) X_test]; %加入Bias

weights = zeros(size(X_train, 2), 1);
lr = 0.001;
epochs = 100;
costs = zeros(epochs, 1);

%模型訓練
disp(['-----Start  Training-----'])
for epoch = 1:epochs
    pred = 1 ./ (1 + exp(-X_train * weights));
    gradient = X_train' * (pred - y_train);

    weights = weights - lr * gradient;
    
    cost = -sum(y_train .* log(pred) + (1 - y_train) .* log(1 - pred));
    if mod(epoch,10) == 0 || epoch==1
        fprintf('Epoch%d Loss: %f\n',epoch,cost);
    end
    costs(epoch) = cost;
end

disp(['-----Finish Training-----'])


%模型預測
disp(['-----Start  Predict-----'])

pred = 1 ./ (1 + exp(-X_test * weights));
prediction = round(pred);

disp(['-----Finish Predict-----'])

%模型結果
acc = sum(prediction == y_test) / length(y_test);
fprintf('Accuracy: %f\n', acc);

confusion_table = confusionmat(y_test, prediction);
disp('Confusion Table for Logistic Regression:');
disp('          Predicted 0   Predicted 1');
disp(['Actual 0  ' num2str(confusion_table(1,1)) '             ' num2str(confusion_table(1,2))]);
disp(['Actual 1  ' num2str(confusion_table(2,1)) '             ' num2str(confusion_table(2,2))]);

%畫圖
figure;
plot(1:epochs, costs);
xlabel('Iteration');
ylabel('Loss');
title('Model Loss');

%匯入資料
data = load("IMU_label_data.mat");

%資料前處理
X_train = data.IMU_label_data(1:100, 1:3);
y_train = data.IMU_label_data(1:100, 4);
X_test = data.IMU_label_data(101:200, 1:3);
y_test = data.IMU_label_data(101:200, 4);
y_train(y_train == 0) = -1;
y_test(y_test == 0) = -1;

%初始化模型參數及訓練參數
num_WeakClassifier = 3;%三個弱模型
N = size(X_train, 1);
D = ones(N, 1) / N;

classifiers = cell(num_WeakClassifier, 1);
alphas = zeros(num_WeakClassifier, 1);

%模型訓練
disp(['-----Start Training-----'])

for t = 1:num_WeakClassifier
    stream = RandStream('mlfg6331_64','Seed',t); %隨機種子碼
    classifier = fitctree(X_train, y_train, 'Weights', D, 'Stream', stream);
    y_pred = predict(classifier, X_train);
    error = sum(D .* (y_pred ~= y_train));

    alpha = 0.5 * log((1 - error) / max(error, eps));
    alphas(t) = alpha;

    D = D .* exp(-alpha * y_train .* y_pred);
    D = max(D / sum(D),eps);

    classifiers{t} = classifier;
end

disp(['-----Finish Training-----'])

%模型預測
disp(['-----Start  Predict-----'])

N_test = size(X_test, 1);
scores = zeros(N_test, 1);

for t = 1:num_WeakClassifier
    classifier = classifiers{t};
    alpha = alphas(t);
    y_pred = predict(classifier, X_test);
    scores = scores + alpha * y_pred;
end

prediction = sign(scores);
prediction(prediction == 0) = 1;

disp(['-----Finish Predict-----'])

%模型結果
acc = sum(prediction==y_test)/length(y_test);
fprintf('Accuracy: %f\n',acc);

confusion_table = confusionmat(y_test, prediction);
disp('Confusion Table for Adaboost:');
disp('          Predicted 0   Predicted 1');
disp(['Actual 0  ' num2str(confusion_table(1,1)) '             ' num2str(confusion_table(1,2))]);
disp(['Actual 1  ' num2str(confusion_table(2,1)) '             ' num2str(confusion_table(2,2))]);

%弱模型權重
disp('Weak Model Weights:');
for t = 1:num_WeakClassifier
    fprintf('Model %d: %f\n', t, alphas(t));
end


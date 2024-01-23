data = load("IMU_label_data.mat");
%data.IMU_label_data
X_train = data.IMU_label_data(1:100,1:3);
y_train = data.IMU_label_data(1:100,4);
X_test  = data.IMU_label_data(101:200,1:3);
y_test  = data.IMU_label_data(101:200,4);
y_train(y_train == 0) = -1;
y_test(y_test == 0) = -1;

weights = zeros(1, size(X_train, 2) + 1);
epochs = 10;
lr = 0.001;
costs = zeros(1, epochs+1);

num_errors = sum(sign([ones(size(X_train, 1), 1), X_train] * weights') ~= y_train);
costs(1) = num_errors;

for epoch = 1:epochs
    num_errors = 0;
    for i = 1:size(X_train, 1)
        xi = [1,X_train(i,:)];
        yi = y_train(i);
        if yi*(weights*xi')<=0
            weights = weights + lr*yi*xi;
            num_errors = num_errors + 1;
        end
    end
    costs(epoch+1) = num_errors;
end

prediction = sign([ones(size(X_test, 1), 1), X_test] * weights');
acc = sum(prediction==y_test)/length(y_test);
fprintf('Accuracy: %f\n',acc);

confusion_table = confusionmat(y_test, prediction);
disp('Confusion Table for Perceptron:');
disp('          Predicted 0   Predicted 1');
disp(['Actual 0  ' num2str(confusion_table(1,1)) '             ' num2str(confusion_table(1,2))]);
disp(['Actual 1  ' num2str(confusion_table(2,1)) '             ' num2str(confusion_table(2,2))]);

figure;
plot(0:epochs, costs);
xlabel('Iteration');
ylabel('Loss');
title('Loss vs. Iteration for Perceptron');

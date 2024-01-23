%匯入資料
xy_data = importdata('xy_data.mat');

xy_data_shape = size(xy_data);
duration = xy_data_shape(3);

% 資料前處理
X_train = [];
Y_train = [];

for t = 1:60

    %Segment
    [Seg, Si_n, S_n] = Segment(xy_data(:,:,t));

    %Label
    PN = Label(t, S_n);

    for i = 1:S_n    
        %提取特定段的資料
        segment_data = xy_data(Seg(1:Si_n(i),i),:,t);
        % 重塑為二維資料（x和y座標）
        segment_data = reshape(segment_data, [], 2);

        %每段有幾點
        n = size(segment_data, 1);

        %每個段的標準偏差
        mean_x = mean(segment_data(:,1));
        mean_y = mean(segment_data(:,2));
        distances = sqrt((segment_data(:,1) - mean_x).^2 + (segment_data(:,2) - mean_y).^2);
        std_deviation = std(distances);

        %每個段的寬度
        width = max(max(segment_data)) - min(min(segment_data));
        
        %圓度和半徑
        A = [-2*segment_data(:,1), -2*segment_data(:,2), ones(size(segment_data, 1), 1)];
        b = [-segment_data(:,1).^2 - segment_data(:,2).^2];
        x_prime = pinv(A' * A) * A' * b;
        xc = x_prime(1);
        yc = x_prime(2);
        rc = sqrt(abs(xc^2 + yc^2 + x_prime(3)));
        distances_to_center = sqrt((segment_data(:,1) - xc).^2 + (segment_data(:,2) - yc).^2);
        sc = sum(abs(rc - distances_to_center));
        
        features = [n, std_deviation, width, sc, rc];
        X_train = [X_train; repmat(features, n, 1)];
        Y_train = [Y_train; PN(i) * ones(n, 1)];
    end
end

% imag_part = imag(X_train);  % 獲取數據的虛數部分
% has_imag = any(imag_part(:) ~= 0);  % 檢查是否有虛數部分不為零的元素
% 
% if has_imag
%     disp('X_train 中包含虛數');
% else
%     disp('X_train 中不包含虛數');
% end


% 模型訓練
weights = zeros(1, size(X_train, 2) + 1);
epochs = 100;
lr = 0.00001;
num_errors = sum(sign([ones(size(X_train, 1), 1), X_train] * weights') ~= Y_train);
costs = zeros(1, epochs+1);
costs(1) = num_errors;
disp(['-----Start  Training-----']);
fprintf('Epoch0 Loss: %f\n',num_errors);
for epoch = 1:epochs
    num_errors = 0;
    for i = 1:size(X_train, 1)
        xi = [1,X_train(i,:)];
        yi = Y_train(i);
        if yi*(weights*xi')<=0
            weights = weights + lr*yi*xi;
            num_errors = num_errors + 1;
        end
    end
    fprintf('Epoch%d Loss: %f\n',epoch,num_errors);
    costs(epoch+1) = num_errors;
end
disp(['-----Finish Training-----']);

% 模型預測
Y_test = [];
Y_pred_all = [];
for t = 61:duration
    clf;
    title([num2str(t), ' sec.']);
    [Seg, Si_n, S_n] = Segment(xy_data(:,:,t));
    PN = Label(t, S_n);
    fprintf(' There are %i segments at %i sec \n', S_n, t);

    hold on;
    plot(xy_data(:,1,t), xy_data(:,2,t), '.');

    for i = 1:S_n
        %提取特定段的資料
        segment_data = xy_data(Seg(1:Si_n(i),i),:,t);

        % 重塑為二維資料（x和y座標）
        segment_data = reshape(segment_data, [], 2);

        %每段有幾點
        n = size(segment_data, 1);

        %每個段的標準偏差
        mean_x = mean(segment_data(:,1));
        mean_y = mean(segment_data(:,2));
        distances = sqrt((segment_data(:,1) - mean_x).^2 + (segment_data(:,2) - mean_y).^2);
        std_deviation = std(distances);

        %每個段的寬度
        width = max(max(segment_data)) - min(min(segment_data));

        %圓度和半徑
        A = [-2*segment_data(:,1), -2*segment_data(:,2), ones(size(segment_data, 1), 1)];
        b = [-segment_data(:,1).^2 - segment_data(:,2).^2];
        x_prime = pinv(A' * A) * A' * b;
        xc = x_prime(1);
        yc = x_prime(2);
        rc = sqrt(abs(xc^2 + yc^2 + x_prime(3)));
        distances_to_center = sqrt((segment_data(:,1) - xc).^2 + (segment_data(:,2) - yc).^2);
        sc = sum(abs(rc - distances_to_center));

        features = [n, std_deviation, width, sc, rc];
        Y_pred = sign([1, features] * weights');
        Y_test = [Y_test; PN(i)];
        Y_pred_all = [Y_pred_all; Y_pred];

        if Y_pred >= 0
            plot(segment_data(:,1), segment_data(:,2), 'o');
        end
    end
    hold off;
    pause(0.2);
end

% 模型結果
confusion_table = confusionmat(Y_test, Y_pred_all);
acc = sum(diag(confusion_table)) / sum(confusion_table(:));
fprintf('Accuracy: %f\n', acc);

disp('Confusion Table for Perceptron:');
disp('          Predicted -1   Predicted 1');
disp(['Actual -1  ' num2str(confusion_table(1,1)) '             ' num2str(confusion_table(1,2))]);
disp(['Actual 1  ' num2str(confusion_table(2,1)) '             ' num2str(confusion_table(2,2))]);

% clear;
% clc;

file_path = 'imu_dataHW3';
fileID = fopen(file_path, 'r');
data = textscan(fileID, '%f', 'Delimiter', '\n');
fclose(fileID);
data = reshape(data{1}, 7, []).';

velocity_x = data(:, 2);
velocity_y = data(:, 3);
velocity_z = data(:, 4);
acc_x = data(:, 5);
acc_y = data(:, 6);
acc_z = data(:, 7);

dt = 0.02;

theta_x_acc = zeros(length(data), 1);

for i = 1:length(theta_x_acc)
    theta_x_acc(i) = atan2(acc_y(i), acc_z(i));
    theta_x_acc(i) = rad2deg(theta_x_acc(i));
    if theta_x_acc(i) > 0
        theta_x_acc(i) = theta_x_acc(i) - 180;
    else
        theta_x_acc(i) = theta_x_acc(i) + 180;
    end
    theta_x_acc(i) = theta_x_acc(i) + 48;
end
theta_x_acc = deg2rad(theta_x_acc);

theta_x_gyro = zeros(length(data), 1);
theta_x_est = zeros(length(data), 1);

Q = 0.001;
R = 0.1;

A = 1;
B = 0.02;
C = 1;

u = deg2rad(velocity_x);
z = theta_x_acc;
x = 0;
estimate_sigma = 1;
for k = 1:length(data)

    x = x + u(k) * B; 
    estimate_sigma = A * estimate_sigma * A' + Q;
    K = estimate_sigma * C' / (C * estimate_sigma * C' + R);
    x = x + K * (z(k) - C * x);
    estimate_sigma = (1 - K * C) * estimate_sigma;
   
    theta_x_est(k) = x;
    theta_x_gyro(k) = theta_x_gyro(max(k-1,1)) + velocity_x(k) * dt;
end

theta_x_est = rad2deg(theta_x_est);
theta_x_acc = rad2deg(theta_x_acc);
theta_x_gyro = cumsum(velocity_x) * dt;


figure;
iterations = 1:length(velocity_x);
plot(iterations, theta_x_gyro, 'g', ...
     iterations, theta_x_acc, 'r', ...
     iterations, theta_x_est, 'b');
legend('\theta_{x} from gyroscope', '\theta_{x} from accelerometer', '\theta_{x} EKF estimate');
xlabel('Iteration');
ylabel('Angle (degrees)');
title('EKF Angle Estimation');
grid on;

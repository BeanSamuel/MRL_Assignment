clc;
clear;
heading_data1 = load('heading_estimation_data1.dat');
gyro_data1 = load('gyro_data1.mat').data; 
heading_data2 = load('heading_estimation_data2.dat');

R = std(gyro_data1)^2;
Q = std(heading_data1(:, 2))^2;
% disp([R,Q])
A = 1;
B = 0.01;
C = 1;

theta_estimates = zeros(length(heading_data2), 1);

u = heading_data2(:, 1);
z = heading_data2(:, 2);

x = 0;
estimate_sigma = 1;

for k = 1:length(heading_data2)

    x = A * x + B * u(k);
    estimate_sigma = A*estimate_sigma*A' + Q;
    K = (estimate_sigma*C') / (C*estimate_sigma*C'+R);
    x = x + K * (z(k) - C*x);
    estimate_sigma = (1 - K*C) * estimate_sigma;

    theta_estimates(k) = x;
end

figure;
hold on;
plot(heading_data2(:, 2), 'r', 'DisplayName', 'Measured zθ');
plot(theta_estimates, 'b', 'DisplayName', 'Estimated θz');
xlabel('Time Step');
ylabel('Angle (rad)');
title('Measured vs Estimated Angle');
legend;
hold off;

% figure;
% plot(heading_data2(:, 1));
% xlabel('Time Step');
% ylabel('Angular Velocity (rad/s)');
% title('Angular Velocity ωz');
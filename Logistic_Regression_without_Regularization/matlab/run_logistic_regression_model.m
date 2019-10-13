clear all
close all
clc

% load data
data = load('Sample Data.txt');
% Grab the relevant data
y = data(:,end);
x = data(:,1:end-1);

%plot Distribution of the data
figure;
plot(x((y==0),1),x((y==0),2),'bo');
hold on
plot(x((y==1),1),x((y==1),2),'ro');

xlabel("Feature x1");
ylabel("Feature x2");
legend("Negative Class","Positive Class")

%normalize x, add ones to x.
x_norm = normalize_features(x);
x_ones = ones(size(x_norm,1),1);
x_norm = [x_ones x_norm];

% Other parameters, initialize zeros for theta
theta = zeros(size(x_norm,2),1);
alpha = 0.1;
iterations = 3000;
% Call the Gradient descent function
[J,theta] = gradient_descent_logistic_regression(theta,x_norm,y,alpha,iterations)

%Plot the Cost Function
figure;
plot(1:iterations, J);
xlabel("Iteration #");
ylabel("Cost Values");
title("Cost Function");

%compute predicted y
h = compute_sigmoid(x_norm*theta);
[overall_accuracy,tp_count,fp_count] = performance_measure(double(h>0.5), y)

%visualize the performance
figure;
plot(x((y==0),1),x((y==0),2),'bo');
hold on
plot(x((y==1),1),x((y==1),2),'ro');
hold on
plot(x((double(h>0.5)==0),1),x((double(h>0.5)==0),2),'b*');
hold on
plot(x((double(h>0.5)==1),1),x((double(h>0.5)==1),2),'r*');

xlabel("Feature x1");
ylabel("Feature x2");
legend("Negative Class - True Label","Positive Class - True Label","Negative Class - Predicted Label","Positive Class - Predicted Label")
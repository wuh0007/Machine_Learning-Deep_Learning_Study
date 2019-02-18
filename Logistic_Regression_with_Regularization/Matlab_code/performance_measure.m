function [overall_accuracy,tp_count,fp_count] = performance_measure(predicted_y, actual_y)
 % performance measurement for logistic regression 
 % total number of samples
 m = length(actual_y);
 % number of samples(h=y)
 c = length(find(predicted_y == actual_y));
 
 %number of samples(h=y)/total number of samples
 overall_accuracy = c/m * 100
 %number of samples(h=y=1)
 tp_count = length(find(actual_y==1 & predicted_y==1))
 %number of samples(h=y=0)
 fp_count = length(find(actual_y==0 & predicted_y==1))
  
end
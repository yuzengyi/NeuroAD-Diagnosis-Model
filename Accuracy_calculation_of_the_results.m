load('neural_network_results.mat')
% predict_X is the input independent variable
Predict_Y=zeros(100,1);
for i =1:100
    Predict_Y(i,1)=sim(net,predict_X(i,:)');
end
Predict_Y(abs(Predict_Y)<=0.5)=0;
Predict_Y(abs(Predict_Y)>0.5)=1;

% Calculate accuracy
accuracy = sum(Predict_Y == preY) / length(Predict_Y);
% Output the result
disp(['Accuracy: ', num2str(accuracy)]);

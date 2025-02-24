close all

clear all
close all

limit=12;
[num,txt,raw] = xlsread('Smartwatch Data');
Fs = 100;            % Sampling frequency                    
T = 1/Fs;             % Sampling period      

% clear all
% close all
% [num,txt,raw] = xlsread('100 Hz 1000 Samples 3 segment.xlsx'); 

age=num(1:end,2:2);
features=num(1:end,5:end);

for i=1:size(features,1) 
%     for i=1:15
signal=features(i,:);
fs=500; %sample rate in kHz
order=4;   %order of filter
% Define the cutoff frequencies (in Hz)
f_low = 4.9;    % Lower cutoff frequency
f_high = 5;    % Higher cutoff frequency
% Normalize the cutoff frequencies with respect to Nyquist frequency
Wn = [f_low f_high] / (fs/2);
% Design a 4th order Butterworth filter
[b, a] = butter(order, Wn, 'bandpass');
filtsig=filter(b,a,signal);  %filtered signal
features_raw(i,:)=filtsig;
end

for i=1:size(features_raw,1) 
 signal=features_raw(i,:);   
[C,L]= wavedec(signal,4,'sym4');
E=appcoef(C,L,'sym4');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuressym(i,:)=[E,d1,d2,d3,d4];
end
% mdl = fitlm(featuressym,age);
% f5=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
signal=features_raw(i,:);
[C,L]=wavedec(filtsig,4,'db1');
E=appcoef(C,L,'db1');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb1(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb1,age);
f1=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
signal=features_raw(i,:);
filtsig=filter(b,a,signal);  %filtered signal
[C,L]=wavedec(filtsig,4,'db2');
E=appcoef(C,L,'db2');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb2(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb2,age);
f2=mdl.Rsquared.ordinary*1e4;


for i=1:size(features_raw,1) 
filtsig=filter(b,a,signal);  %filtered signal
[C,L]= wavedec(filtsig,4,'db3');
E=appcoef(C,L,'db3');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb3(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb3,age);
f3=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
signal=features_raw(i,:);
filtsig=filter(b,a,signal);  %filtered signal
[C,L]= wavedec(filtsig,4,'db4');
E=appcoef(C,L,'db4');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb4(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb4,age);
f4=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
filtsig=features_raw(i,:);
[C,L]= wavedec(filtsig,4,'db5');
E=appcoef(C,L,'db5');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb5(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb5,age);
f5=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
filtsig=features_raw(i,:);
[C,L]= wavedec(filtsig,4,'db6');
E=appcoef(C,L,'db6');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb6(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb6,age);
f6=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
filtsig=features_raw(i,:);
[C,L]= wavedec(filtsig,4,'db7');
E=appcoef(C,L,'db7');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb7(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb7,age);
f7=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
filtsig=features_raw(i,:);
[C,L]= wavedec(filtsig,4,'db8');
E=appcoef(C,L,'db8');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb8(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb8,age);
f8=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
filtsig=features_raw(i,:);
[C,L]= wavedec(filtsig,4,'db9');
E=appcoef(C,L,'db9');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb9(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb9,age);
f9=mdl.Rsquared.ordinary*1e4;

for i=1:size(features_raw,1) 
filtsignal=features_raw(i,:);
[C,L]= wavedec(filtsig,4,'db10');
E=appcoef(C,L,'db10');
[d1,d2,d3,d4] = detcoef(C,L,[1 2 3 4]);
featuresdb10(i,:)=[E,d1,d2,d3,d4];
end
mdl = fitlm(featuresdb10,age);
f10=mdl.Rsquared.ordinary*1e4;

f=0;
w1=f1/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w2=f2/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w3=f3/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);;
w4=f4/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w5=f5/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w6=f6/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w7=f7/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w8=f8/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w9=f9/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w10=f10/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);
w=f/(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f);

w=1;
w1=1;
w2=1;
w3=1;
w4=1;
w5=1;
w6=1;
w7=1;
w8=1;
w9=1;
w10=1;


for i=1:size(features_raw,1)
    table(:,:,i)= [
    featuresdb1(1:950);
    featuresdb2(1:950);
    featuresdb3(1:950);
    featuresdb4(1:950);
    featuresdb5(1:950);
    featuresdb6(1:950);
    featuresdb7(1:950);
    featuresdb8(1:950);
    featuresdb9(1:950);
    featuresdb10(1:950);
    ]
end

table=cat(2,featuresdb1*w1,featuresdb2*w2,featuresdb3*w3,featuresdb4*w4,featuresdb5*w5,featuresdb6*w6,featuresdb7*w7,featuresdb8*w8,featuresdb9*w9,featuresdb10*w10);
table=features_raw;

% kernel1=[-32:32];
% increment1=size(kernel1,2);
% for s=1:size(table,1)
%     input1=table(s,:) ;
% res1(:,:)=conv(kernel1,input1);
% i=1;
% j=1;
% k=1;
% for k=1:(size(res1,2)/increment1)
%     A1=res1(i,j:j+increment1-1);
% maxpool1(k)=max(A1);
% k=k+1;
% j=j+increment1;
% end
% output1(s,:)=(maxpool1-std(maxpool1))/mean(maxpool1);
% end

table=output1;
%table=cat(2,featuresdb4*w1,featuresdb7*w4,featuresdb10*w3,featuresdb9*w2,featuresfreq*w6,featuresfreq*w5);
% % table=cat(2,featuresdb4*w1,featuresfreq*w6,featuressym*w5);
% 
% % table=awgn(table,20);
% % 
% % while(count<2)
% 
% close all

numFeatures = size(table,2);
numResponses = 1;
numHiddenUnits = 1000;

XTrain = (table(1:43,1:end));
YTrain = (age(1:43,1:end));
XTest= (table(44:end,1:end));
YTest = (age(44:end,1:end));

XTes = transpose(XTest);
YTes = transpose(YTest );
XTrai= transpose(XTrain);
YTrai = transpose(YTrain);

for i=1:size(YTrai,2)
    if (YTrai(i)<limit) 
        YTrai(i)=1;
    else
        YTrai(i)=2;
    end
end

for i=1:size(YTes,2)
    if (YTes(i)<limit) 
        YTes(i)=1;
    else
         YTes(i)=2;
    end
end
% inputSize = [500 1];  % Assuming 500 time steps, 1 channel (time-series data)
% numClasses = 5;       % Number of output classes
% embeddingDim = 128;   % Dimensionality of token embeddings
% numHeads = 8;         % Number of attention heads
% ffnDim = 512;         % Dimensionality of feed-forward network
% numLayers = 6;        % Number of transformer encoder layers
% 
% function layers = transformerEncoderLayer(embedDim, numHeads, ffnDim, blockName)
%     layers = [
%         Multi-head attention layer
%         multiheadAttentionLayer(numHeads, embedDim, 'Name', [blockName, '_multihead_attention'])
% 
%         Add & Norm layer 1
%         additionLayer(2, 'Name', [blockName, '_add1'])
%         layerNormalizationLayer('Name', [blockName, '_norm1'])
% 
%         Feed-forward network
%         fullyConnectedLayer(ffnDim, 'Name', [blockName, '_fc1'])
%         reluLayer('Name', [blockName, '_relu'])
%         fullyConnectedLayer(embedDim, 'Name', [blockName, '_fc2'])
% 
%         Add & Norm layer 2
%         additionLayer(2, 'Name', [blockName, '_add2'])
%         layerNormalizationLayer('Name', [blockName, '_norm2'])
%     ];
% end
% 
% layers = [ ...
%   sequenceInputLayer(numFeatures)
% flattenLayer
 lstmLayer(numHiddenUnits)
fullyConnectedLayer(numResponses)
 regressionLayer];
% residualBlock(256, 3, 2, 'residual_block_3')
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.01, ...
    'Verbose',false, ...
    'Plots','training-progress');

net= trainNetwork(XTrai,YTrai,layers,options);
test_outcome=round(predict(net,XTes));

multiplier=1;
predicted=test_outcome*multiplier;
actual=YTest;
scatter(actual,predicted);
E=predicted-actual;
MAE= mae(E,predicted,actual);

% n=size(test_outcome,2);
% Actual=YTes;
% Predicted=test_outcome;
% % E=Actual-Predicted;
% AE=abs(E);
% MAE=sum(AE)/n;

% for i=1:size(Actual,2)
%    temp(i)= AE(i)-MAE;
% end
% SD= sqrt(sum(temp.^2)/(n-1));
% CI=MAE+1.96*SD/sqrt(n);

% for i=1:size(YTes,2)
%     if YTes(i)<limit 
%         actualclass(i)=1;
%     else
%          actualclass(i)=2;
%     end
% end
% 
% for i=1:size(test_outcome,2)
%     if test_outcome(i)<limit 
%        predictedclass(i)=1;
%     else
%          predictedclass(i)=2;
%     end
% end

predictedclass=test_outcome;
actualclass=YTes;
Accuracy=(size((find(test_outcome==YTes)),2)/size(YTes,2))*100
TP=sum((predictedclass == 2) & (actualclass == 2));
TN=sum((predictedclass == 1) & (actualclass == 1));
FP=sum((predictedclass == 2) & (actualclass == 1));
FN=sum((predictedclass == 1) & (actualclass == 2));

TPR=TP/(TP+FN);
TNR=TN/(FP+TN);
FPR=FP/(FP+TN);
FNR=FN/(FP+TN);
% Accuracy=(TP+TN)/(TP+TN+FP+FN);
PPV=TP/(TP+FP);
NPV=TN/(TN+FN);
FDR=FP/(FP+TP);
FOR=FN/(FN +TN);
LRp=TPR/FPR;
LRn=FNR/TNR;
F1=(2*PPV*TPR)/(TPR+PPV);
Matrix=[Accuracy TPR TNR FPR FNR PPV NPV FDR FOR LRp LRn F1];
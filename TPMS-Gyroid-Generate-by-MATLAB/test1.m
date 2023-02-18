% finished=[];
% save finished.mat finished;
num=20;
pre='Ti_BO_R10_';
address=['E:/daijiabao/abaqus1000randomTi/BO_R10/inp' '/'];
tic
for l=1:num
    fprintf('进度是：%6.4f  ',l/num);
    toc
    random(l,pre,address);
end

% Exiting Automatically
% exit
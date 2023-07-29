%finished=[];
%save finished.mat finished;
num=5000;
tic
for l=1:num
    fprintf('进度是：%6.4f  ',l/num);
    toc
    random(l);
end

% Exiting Automatically
exit
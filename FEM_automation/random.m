function ramdom(l);
%% 
load finished.mat;
%% 
n=3;
r1=zeros(n^3,3);
for a=1:n
    for b=1:n
        for c=1:n
            r1(n*n*a-n*n+n*b-n+c-1+1,:)=[a,b,c];
        end
    end
end
r2OK=0;
[~,w]=size(finished);
alarm=0;
while(r2OK==0)
r2OK=1;
alarm=alarm+1;
if(alarm>=2)
    fprintf('重复\n');
end
%r2=randi(6,n*n*n,1);
r2=6*ones(27,1);
for u=1:w
    if(r2==finished(:,u))
        r2OK=0;
    end
end
end
finished=[finished r2];
save finished.mat finished;
data0=[r1 (r2+3)];
%data0(:,4)=9;
%% 
nofgyroid=1;
E_v=[9 1+1.0736;
     8 1+0.7909;
     7 1+0.4931;
     6 1+0.2254;
     5 1-0.0573;
     4 1-0.3400;];
E_v2=E_v(:,2);
for i=1:n^3
data0(i,4)=E_v2(E_v(:,1)==data0(i,4));
end
%% 
b=3;
sizeofdata0=[n,n,n];
accu=21;
v=createv_2(data0,sizeofdata0,accu,b);
%% 
sizeofgyroid=2;
stldataofgyroid=drawgyroid(v,sizeofdata0,accu,sizeofgyroid,sizeofgyroid*data0-1);
name=num2str(l);
name=[name '.stl'];
stlwrite(l,name,stldataofgyroid);

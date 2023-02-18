function random(l,pre,address);
%% 
load Ti_BO_R10_finished.mat;
%%
n1=3;
n2=3;
n3=3;
r1=zeros(n1*n2*n3,3);
for a=1:n1
    for b=1:n2
        for c=1:n3
            r1(n2*n3*a-n2*n3+n3*b-n3+c-1+1,:)=[a,b,c];
        end
    end
end
r2=finished(:,l);
r2=round(10*(1-r2));
data0=[r1 r2];
%% 
nofgyroid=1;
E_v=[9 1+1.0736;
    88888 2.17
    863 1+0.98;
    85 1+0.93225;
     8 1+0.7909;
     7 1+0.4931;
     6 1+0.2254;
     55 ((1+0.2254)+(1-0.0573))/2;
     500 1+0.120801
     5 1-0.0573;
     1000 1-0.31173
     4 1-0.3400;
     3 1-0.6227;
     2 1-0.9054;
     1001 0.9+0.5625
     1002 0.9+1.0067
     1003 0.9+1.2512
     1004 0.9+1.0169
     94 2.2013
     96 2.2578
     98 2.3000
     99 2.3426
     20 0.1091
     50 0.9573
     80 1.80545 
     4000 1.55266
     ];
E_v2=E_v(:,2);
for i=1:n1*n2*n3
data0(i,4)=E_v2(E_v(:,1)==data0(i,4));
end
%% 
b=3;
sizeofdata0=[n1,n2,n3];
accu=21;
v=createv_2(data0,sizeofdata0,accu,b);
theV=2;
anoV=2-(l-1)*0.2;
v1=anoV*ones(n1*accu,n1*accu,119);
for kk=1:round((n3*accu-2*119-1)/2)-8-34
    the_num=round((n3*accu-2*119-1)/2-8)-34+1;
    derta=(theV-anoV)/the_num;
    vnow=(anoV+kk*derta)*ones(n1*accu,n1*accu,1);
    v1=cat(3,v1,vnow);
end
v1=cat(3,v1,theV*ones(n1*accu,n1*accu,17+34+34));
for kkk=1:round((n3*accu-2*119-1)/2-8)-34
    the_num=round((n3*accu-2*119-1)/2-8)-34+1;
    derta=(anoV-theV)/the_num;
    vnow=(theV+kkk*derta)*ones(n1*accu,n1*accu,1);
    v1=cat(3,v1,vnow);
end
v1=cat(3,v1,anoV*ones(n1*accu,n1*accu,119));
% v=v1;

%% 
sizeofgyroid=2;
stldataofgyroid=drawgyroid(v,sizeofdata0,accu,sizeofgyroid,sizeofgyroid*data0-1);
name=[num2str(l) '.stl'];
name=[pre name];
filenm = [num2str(l) '.txt'];
filenm = [pre filenm];
stlwrite(filenm,address,name,stldataofgyroid);
end

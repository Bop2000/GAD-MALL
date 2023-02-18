randomTi1000_cell=cell(1000,1);
for i=1:1000
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
r2=finished(:,i);
data0=[r1 r2];
a=zeros(3,3,3);
for j=1:27
    a(data0(j,2),data0(j,1),data0(j,3))=data0(j,4);
end
randomTi1000_cell{i,1}=a;
end

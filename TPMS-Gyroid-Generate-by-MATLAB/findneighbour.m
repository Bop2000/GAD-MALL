function neighbourhoods=findneighbour(inputdata,position)
neighbourhoods=zeros(3,3,3);
neighbourhoods(:,:,:)=NaN;
[r,~]=size(inputdata);
flag=0;
for i=1:r
    if ((inputdata(i,1)==position(1))&(inputdata(i,2)==position(2))&(inputdata(i,3)==position(3)))
        flag=1;
        id=i;
    end
end
if flag~=0
    for i=1:r
        dertax=inputdata(i,1)-position(1);
        dertay=inputdata(i,2)-position(2);
        dertaz=inputdata(i,3)-position(3);
        if ((abs(dertax))<=1&(abs(dertay))<=1&(abs(dertaz))<=1)
            neighbourhoods(dertax+2,dertay+2,dertaz+2)=inputdata(i,4);
        end
    end
end
end
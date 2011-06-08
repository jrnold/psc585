
function P=Ptilde(Pp,Pg,model)
%Pp is a k*2^k by 2*k matrix with the probability of the i-th province
%choosing to revolt or not in the (i-1)*2+(1:2) columns
%Pg is a k*2^k by k matrix with the probability of g choosing to send army
%to k-th province
n=model.n;
k=model.k;
m=model.m;
P=zeros(n,n);

for i=1:k
    for sp=1:m
        if model.S(sp,i)==0
            A=Pp(:,(1:2:(2*k-1))+model.S(sp,:));
            A(:,i)=1;
            P(:,(i-1)*m+sp)=bsxfun(@times,Pg(:,i),prod(A,2));
        end
    end
end
end
function [V]=Phigprov(Pp,Pg,theta,model)
n=model.n;
k=model.k;
m=model.m;
x=model.x;
y=model.y;
delta=model.delta;

B=zeros(n,k+1);
P=Ptilde(Pp,Pg,model);

Zg=zeros(n,5);
Zg(:,1)=sum(Pp(:,(0:2:(2*(k-1)))+1),2);
Zg(:,2)=sum(bsxfun(@times,Pp(:,(0:2:(2*(k-1)))+1),x),2);
for l=1:k
    Zg((l-1)*m+(1:m),3)=-sum(bsxfun(@times,Pg((l-1)*m+(1:m),:),model.D(l,:)),2);
end
Zg(:,4)=-model.wg;
Zg=Zg*model.g1oversigma;
Eg=sum(bsxfun(@times,Pg,-log(Pg)),2);

Zi=ones(n,5);
for i=1:k
   Ei=bsxfun(@times,Pp(:,(i-1)*2+1),-log(Pp(:,(i-1)*2+1)))+...
       bsxfun(@times,Pp(:,(i-1)*2+2),-log(Pp(:,(i-1)*2+2)));
   
   Zi(:,2)=x(i);
   Zi(:,3)=-(1-model.S(:,i));
   Zi(:,4)=-Pg(:,i);
   Zi(:,5)=-Pg(:,i)*y(i);
   Zi=bsxfun(@times,Zi,Pp(:,(i-1)*2+2));
   B(:,i)=[Zi Ei]*[theta;1];
end
theta(3)=1;
B(:,k+1)=[Zg Eg]*[theta;1];

%The values are stored in the n x (k+1) matrix V; the first k columns
%correspond to the k provinces, the k+1 to the government.
V=(eye(n)-delta*P)\B;
end


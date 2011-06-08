function [Pp Pg]=NewP(Pp,Pg,theta,model)
n=model.n;
k=model.k;
m=model.m;
x=model.x;
y=model.y;
delta=model.delta;
V=Phigprov(Pp,Pg,theta,model);
vp=zeros(n,2*k);
vg=zeros(n,k);
thetag=theta;
thetag(3)=1;
thetag(4)=1;
Z1i=ones(n,5);
Zag=zeros(n,5);
for i=1:k
    Ppi=Pp;
    Ppi(:,(i-1)*2+(1:2))=[ones(n,1),zeros(n,1)];
    Pi=Ptilde(Ppi,Pg,model);
    vp(:,(i-1)*2+1)=delta*Pi*V(:,i);
    
    Z1i(:,2)=x(i);
    Z1i(:,3)=-(1-model.S(:,i));
    Z1i(:,4)=-Pg(:,i);
    Z1i(:,5)=-Pg(:,i)*y(i);
    Ppi(:,(i-1)*2+(1:2))=[zeros(n,1),ones(n,1)];
    Pi=Ptilde(Ppi,Pg,model);
    vp(:,(i-1)*2+2)=Z1i*theta+delta*Pi*V(:,i);
    
    Zag(:,1)=sum(Pp(:,(0:2:(2*(k-1)))+1),2);
    Zag(:,2)=sum(bsxfun(@times,Pp(:,(0:2:(2*(k-1)))+1),x),2);
    for l=1:k
        Zag((l-1)*m+(1:m),3)=-model.D(l,i);
    end
    Zag(:,4)=-model.wg;
    Zag=Zag*model.g1oversigma;
    Pgi=zeros(n,k);
    Pgi(:,i)=ones(n,1);
    Ptilg=Ptilde(Pp,Pgi,model);
    vg(:,i)=Zag*thetag+delta*Ptilg*V(:,k+1);
end

%Compute the new conditional probabilities using the logit formulas
vp=exp(vp);
for i=1:k
   Pp(:,(i-1)*2+(1:2))=bsxfun(@rdivide,vp(:,(i-1)*2+(1:2)),sum(vp(:,(i-1)*2+(1:2)),2)); 
end
vg=exp(vg);
Pg=bsxfun(@rdivide,vg,sum(vg,2));
end
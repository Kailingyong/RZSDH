function [B, P1, P2, t1, t2] = RZSDH(X1, X2, Y, param, A)


X1 = X1'; X2=X2'; Y = Y';
[dx1, n ] = size(X1); [dx2, ~ ] = size(X2); [c, ~] = size(Y);

alphe1 = param.alphe1;
alphe2 = param.alphe2;
beta1 = param.beta1;
beta2 = param.beta2;
lambda = param.lambda;
thea = param.thea;
gamma = param.gamma;
mu = param.mu;

mu1 = param.mu1;
lambda_c = param.lambda_c;

etaX = param.etaX;
etaY = param.etaY;
rho = param.rho;

a = (c*(c+2)+c*sqrt(c*(c+2)))/4+eps; %More precise value can be experimentally obtained.

r = param.nbits;


%% 初始化
D = zeros(size(Y));
Yn = Y;
F1 = zeros(size(Y));
K = zeros(size(Y));
Y1 = zeros(size(Y));


sel_sample = X1(:,randsample(n, 500),:);
[pcaW, ~] = eigs(cov(sel_sample'), r);
V1_c = pcaW'*X1;
sel_sample = X2(:,randsample(n, 500),:);
 [pcaW, ~] = eigs(cov(sel_sample'), r);
V2_c = pcaW'*X2;
V =  (V1_c+V2_c)/2 ;

B = sign(V);
B(B==0)=-1;

t1 = ones(r,1);
t2 = ones(r,1);
en = ones(1,n);
P1 = rand(r,c);
P2 = rand(r,c);


%%
AA = A'*A;
YY = Yn*Yn';
X1X1 = X1*X1';
X2X2 = X2*X2'; 



 for iter = 1:param.iter 
     
      %% B
      B = sign(lambda*V + (r*thea)/(a+etaX+etaY)*(a*V*Yn'*Yn+etaX*V*X1'*X1+etaY*V*X2'*X2));
      B(B==0)=-1;
     
     %% Z1 and Z2
      Z1 = pinv(alphe1*AA)*(alphe1*A'*X1*Yn')*pinv(YY);
      Z2 = pinv(alphe2*AA)*(alphe1*A'*X2*Yn')*pinv(YY);
      
     %% Yn
      Yn = pinv(alphe1*Z1'*A'*A*Z1+alphe2*Z2'*A'*A*Z2+mu1*eye(c))*(rho/2*(Y-D+1/rho*F1)+alphe1*Z1'*A'*X1+alphe2*Z2'*A'*X2+(mu1*K-Y1));
      
      %% K
      [U1,S1,V1] = svd(Yn + Y1./mu1,'econ');
      a1 = diag(S1)-lambda_c/mu1;
      a1(a1<0)=0; 
      T = diag(a1);
      K = U1*T*V1'; 
     
      %% D
      Etp = Y - Yn + 1/rho*F1;
      D = sign(Etp).*max(abs(Etp)- mu/rho,0); 
      
      %% F1
      F1 = F1 + rho*(Y - Yn -D);
      rho = min(1e4, 1.3*rho);
     
      %% P1 and P2
      P1 = (beta1*V-beta1*t1*en)*X1'*pinv(beta1*X1X1+gamma*eye(dx1));
      P2 = (beta2*V-beta2*t2*en)*X2'*pinv(beta2*X2X2+gamma*eye(dx2));
      
      %% t1 and t2
      t1 = (1/n)*(V-P1 *X1)*en';
      t2 = (1/n)*(V-P2 *X2)*en';
      
      
      %% V
      V_1 = (beta1*(P1*X1+t1*en)+beta2*(P2*X2+t2*en)+(r*thea)/(a+etaX+etaY)*(a*B*Yn'*Yn+etaX*B*X1'*X1+etaY*B*X2'*X2)+lambda*B);
      V = Orthogonal_V(V_1,n,r,B,param);
     
     
      %% Y1
      mu1 = 1.01*mu1;
      Y1 = Y1 + mu1*(Yn-K);
      
      
 end

end


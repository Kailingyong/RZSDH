function [V] = Orthogonal_V(Z1,n,nbits,B,param)
     Z = Z1';
     Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-4);
     Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
     Pt = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,nbits-length(find(idx==1))));
     V1 = sqrt(n)*[Pt P_]*[Q Q_]';
     V = pinv(param.thea*B*B'+param.gamma*eye(nbits))*Z1;
end


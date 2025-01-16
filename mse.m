h =figure;
X=imread('a.png'); 
Y=imread('b.png');  
subplot(121);imshow(X);
subplot(122);imshow(Y);
D=X-Y;
MSE = sum(D(:).*D(:))/prod(size(X));
annotation(h,'textbox',[0.4 0.8 0.2 0.05],'String',['MSE=' num2str(MSE)]);
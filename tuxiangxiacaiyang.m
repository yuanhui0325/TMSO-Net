clear all  
a=imread('E:\code\input_Cam036.png');
[line,row]=size(a);%��ȡͼ������ %���²���forѭ������һ����� %b(1:2:line,1:2:row);
L=1; R=1;  
%4��������  
for i=1:2:line; 
    for j=1:2:row;          
        b1(L,R)=a(i,j);          
        R=R+1;%ȡԭͼ��i����һ�е�Ԫ�ظ�����ͼ��Ķ�Ӧλ��     
    end
    L=L+1;%����     
     R=1;%�ӻ��к������ĵ�һ��Ԫ�ؿ�ʼȡԪ�� 
end
figure;
imshow(a); 
title('ԭͼ');%��ʾԭͼ��  
figure;
imshow(b1);
title('4������ͼ');%��ʾ�������ͼ��
imwrite(b1, 'E:\ͼƬ\cameraman������1.tif'); 
%16�������� 
K=1;M=1  
for i=1:4:line;   
    for j=1:4:row;          
        b2(K,M)=a(i,j);          
        M=M+1;%ȡԭͼ��i����һ�е�Ԫ�ظ�����ͼ��Ķ�Ӧλ��     
    end
     K=K+1;%����      
     M=1;%�ӻ��к������ĵ�һ��Ԫ�ؿ�ʼȡԪ�� 
end
figure  
imshow(b2);
title('16������ͼ');%��ʾ�������ͼ��
imwrite(b2, 'E:\ͼƬ\cameraman������2.tif'); 
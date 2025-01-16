clear all  
a=imread('E:\code\input_Cam036.png');
[line,row]=size(a);%读取图像像素 %以下采样for循环可用一句代替 %b(1:2:line,1:2:row);
L=1; R=1;  
%4倍减采样  
for i=1:2:line; 
    for j=1:2:row;          
        b1(L,R)=a(i,j);          
        R=R+1;%取原图像i列下一行的元素赋给新图像的对应位置     
    end
    L=L+1;%换列     
     R=1;%从换列后的列里的第一个元素开始取元素 
end
figure;
imshow(a); 
title('原图');%显示原图像  
figure;
imshow(b1);
title('4倍采样图');%显示采样后的图像
imwrite(b1, 'E:\图片\cameraman采样后1.tif'); 
%16倍减采样 
K=1;M=1  
for i=1:4:line;   
    for j=1:4:row;          
        b2(K,M)=a(i,j);          
        M=M+1;%取原图像i列下一行的元素赋给新图像的对应位置     
    end
     K=K+1;%换列      
     M=1;%从换列后的列里的第一个元素开始取元素 
end
figure  
imshow(b2);
title('16倍采样图');%显示采样后的图像
imwrite(b2, 'E:\图片\cameraman采样后2.tif'); 
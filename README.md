## Keras-Color
Superpixel을 활용한 Color Detection과 GAN을 사용하지 않은 간단한 Art-Style-Transfer 코드를 담고 있다.

## Art Style Transfer
코드는 https://harishnarayanan.org/writing/artistic-style-transfer/를 참조하였다.  
Content Loss와 Style Loss를 효과적으로 줄이는 것이 구현의 핵심이며, 더욱 다양한 변화를 위해서는 GAN을 사용해야 한다.  

## Color Detection
자동차의 색깔을 구분하여 아웃풋을 반환하는 파일인데, 다른 프로젝트에서 사용하기 위해 시연용으로 제작하였다.  
구조는 단순한 CNN인데, 마지막에 아웃풋을 출력하기 전 Superpixel이란 개념을 도입하여 적용하는 것이 핵심이다.  

Superpixel에 관한 개념에 관해서는 https://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-introduction/  
위 사이트에서 예시를 확인할 수 있다.


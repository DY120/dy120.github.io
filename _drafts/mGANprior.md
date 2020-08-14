---
layout: post
title: "Image Processing Using Multi-Code GAN Prior"
tags: [Paper Review, GAN, Generative Model]
comments: true
---

&nbsp;&nbsp;&nbsp;&nbsp;GAN(Generative Adversarial Networks)은 2014년에 발표된 이후 지금까지 많은 연구를 통해 눈부신 발전을 이루었습니다. 특히 무작위로 생성된 latent vector로부터 가상의 이미지를 합성해내는 생성 모델 분야에 큰 영향을 끼쳤는데, 2020년 현재에 이르러서는 BigGAN이나 StyleGAN과 같은 모델을 통해 만들어진 고해상도의 이미지들은 심지어 사람의 눈으로도 가짜라는 걸 분간할 수 없을 만큼 고도화가 이루어졌습니다.

&nbsp;&nbsp;&nbsp;&nbsp;GAN은 이와 같은 강력한 이미지 생성 능력을 기반으로 SISR(Single Image Super-Resolution), denoising, inpainting와 같은 여러 이미지 처리 분야에도 널리 쓰이고 있습니다. 하지만 대부분의 GAN을 이용한 이미지 처리 모델들은 해당 task에만 특화된 loss function과 네트워크 구조를 이용한다는 한계점이 있습니다. 물론 많은 task에 활용할 수 있는 pix2pix와 CycleGAN과 같은 image-to-image translation 모델들이 있긴 하지만, 이러한 모델들 역시 adversarial learning의 개념만 추가한 지도 학습을 진행한다는 한계가 있습니다. 쉽게 말해 GAN의 향만 첨가한 정도라는 것입니다. 그래서 결과물이 생성 모델이 만들어 내는 가상의 이미지만큼의 높은 퀄리티를 가지지 못한다는 단점이 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;최근에는 BigGAN이나 StyleGAN과 같이 미리 학습된 생성 모델을 이용해 GAN의 강력한 이미지 생성 능력을 활용하면서 별도의 학습 또한 필요로 하지 않는 장점을 취하는 방법론을 주로 사용하는 추세입니다. 이를 위해선 기본적으로 실제 이미지로부터 그에 해당하는 latent vector를 출력하는 GAN inversion이라는 기술이 필요합니다. 이는 입력 이미지와 latent vector 간의 reconstruction loss를 정의한 뒤 해당 loss를 최소화하는 latent vector를 찾는 방식인데, 주로 back-propagation이나 추가적인 encoder가 사용되었습니다.

&nbsp;&nbsp;&nbsp;&nbsp;[이전 포스트](https://dy120.github.io/PULSE)에서 살펴 보았던 PULSE 또한 GAN inversion을 활용한 모델이라고 할 수 있습니다. PULSE에서는 결과물을 downscaling 시킨 이미지와 입력된 저화질 이미지 사이의 reconstruction loss를 정의한 뒤 StyleGAN의 latent space에서 back-propagation을 통해 최적의 latent vector를 탐색하는 방식으로 쓰였습니다. 



와 마찬가지로, 이미지 처리에 생성 모델을 활용하려면 실제 이미지를 입력 받아서 그와 연관된 특정 조건을 만족하는 latent vector를 출력할 수 있어야 합니다. 이 과정은 GAN의 역변환으로 볼 수 있기 때문에 주로 GAN inversion이라고 따로 불리며, 그만큼 많은 주목을 받고 있는 task입니다.

입력 이미지와 연관된 reconstruct loss를 최소화하는 방향으로 latent space를 탐색할 수 있어야 합니다. 이 과정은 GAN inversion이라고 따로 불릴 정도로 많은 주목을 받는 task입니다. 실제 이미지를 그와 연관된 특정 조건을 만족하는 latent vector로 역변환하는 과정으로 볼 수 있기 때문에 이러한 이름이 붙게 되었습니다. back-propagation이나 추가적인 encoder를 활용하는 방법들이 제시되었지만 큰 성과를 이루지는 못했습니다.

latent space에서 주어진 조건에 맞는 적절한 latent vector를 찾은 뒤 이를 미리 학습된 GAN 기반 생성 모델에 입력하여 결과물을 얻는 방식입니다. 

연구들에선 latent vector를 입력으로 받아 이미지를 출력한다는 framework를 뒤집어 back-propagation이나 추가적인 encoder를 통해 실제 이미지를 latent vector로 역변환하는 방법을 사용해왔지만 큰 성과를 이루지는 못했습니다.

&nbsp;&nbsp;&nbsp;&nbsp;이번 포스트에서는 CVPR'20에서 소개된 논문인 [**Image Processing Using Multi-Code GAN Prior**](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Image_Processing_Using_Multi-Code_GAN_Prior_CVPR_2020_paper.html)가 제시하는 새로운 접근법인 mGANprior을 살펴보고, 이를 활용해 GAN을 이용한 효과적인 이미지 처리가 어떻게 이루어지는지 알아보도록 하겠습니다.


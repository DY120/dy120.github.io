---
layout: post
title: "Image Processing Using Multi-Code GAN Prior"
tags: [Paper Review, GAN, Generative Model]
comments: true
---

&nbsp;&nbsp;&nbsp;&nbsp;GAN(Generative Adversarial Networks)은 2014년에 발표된 이후 지금까지 많은 연구를 통해 눈부신 발전을 이루었습니다. 특히 무작위로 생성된 latent code로부터 가상의 이미지를 합성해내는 생성 모델 분야에 큰 영향을 끼쳤는데, 2020년 현재에 이르러서는 BigGAN이나 StyleGAN과 같은 모델을 통해 만들어진 고해상도의 이미지들은 심지어 사람의 눈으로도 가짜라는 걸 분간할 수 없을 만큼 고도화가 이루어졌습니다.

&nbsp;&nbsp;&nbsp;&nbsp;GAN은 이와 같은 강력한 이미지 생성 능력을 기반으로 SISR(Single Image Super-Resolution), denoising, inpainting와 같은 여러 이미지 처리 분야에도 널리 쓰이고 있습니다. 하지만 대부분의 GAN을 이용한 이미지 처리 모델들은 해당 task에만 특화된 loss function과 네트워크 구조를 이용한다는 한계점이 있습니다. 물론 많은 task에 활용될 수 있는 pix2pix나 CycleGAN과 같은 image-to-image translation 모델들이 있긴 하지만, 이러한 모델들 역시 adversarial learning의 개념만 추가한 지도 학습을 진행한다는 한계가 있습니다. 다시 말해 GAN의 향만 첨가한 정도라는 것입니다. 그래서 image-to-image translation 모델들이 만드는 결과는 생성 모델이 만들어 내는 가상의 이미지만큼 높은 퀄리티를 가지지 못한다는 단점이 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;이러한 한계점을 극복하고자 최근에는 미리 학습된 BigGAN이나 StyleGAN과 같은 생성 모델을 이용해 GAN의 강력한 이미지 생성 능력을 활용함과 동시에 별도의 학습 또한 필요로 하지 않는 장점을 취하는 방법론을 주로 채용하는 추세입니다. [이전 포스트](https://dy120.github.io/PULSE)에서 살펴본 PULSE라는 모델 또한 이러한 아이디어를 채용하여 개발된 모델이라고 볼 수 있습니다. 

&nbsp;&nbsp;&nbsp;&nbsp;간단하게 설명하자면, 원하는 이미지 처리 결과를 미리 학습된 생성 모델이 만들어 낼 수 있는 이미지의 범위, 즉 latent space 내에서 직접 찾아낸다는 것입니다. 예를 들어서 PULSE는 생성 모델이 만들어낼 수 있는 수없이 많은 얼굴 이미지들 중에서 downscaling을 해서 기존의 입력 이미지가 될 수 있는 이미지들을 back-propagation을 통한 latent code의 최적화를 통해 찾아냄으로써 face hallucination(얼굴에 대한 SISR)을 수행하게 됩니다.

<!---
 PULSE는 입력된 저화질 이미지 $$I_{LR}$$과 downscaling 된 고화질 이미지 $$DS(G(z))$$ 사이의 pixel-wise loss $$\lVert DS(G(z)) - I_{LR} \rVert_p^p$$를 최소화시키는 latent code $$z$$를 back-propagation을 통해 얻어내 SISR을 수행하는 모델이었습니다. 
--->

&nbsp;&nbsp;&nbsp;&nbsp;PULSE는 이와 같은 원리로 꽤 괜찮은 결과를 냈지만, 실제로는 back-propagation을 통해 그 정도로 좋은 결과를 내기는 쉽지 않습니다. PULSE는 latent code를 초기화할 때 Gaussian prior를 걺으로써 latent space를 구의 평면으로 재정의하여 탐색을 진행한다는 새로운 아이디어를 통해 성능 향상을 이룰 수 있었지만, 이와 같은 특정 기술 없이 back-propagation만 사용한 기존의 모델들은 좋은 성능을 내지 못했습니다.

&nbsp;&nbsp;&nbsp;&nbsp;이번 포스트에서 소개해드릴 논문은 이와 같은 문제점이 단일 latent code만을 최적화의 대상으로 삼는 것에서 비롯됐다고 주장하며 새로운 접근법인 mGANprior를 제시했습니다. CVPR'20에서 소개된 논문인 [**Image Processing Using Multi-Code GAN Prior**](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Image_Processing_Using_Multi-Code_GAN_Prior_CVPR_2020_paper.html)를 살펴보면서 이를 활용해 GAN을 이용한 효과적인 이미지 처리가 어떻게 이루어지는지 알아보도록 하겠습니다.

# Introduction

&nbsp;&nbsp;&nbsp;&nbsp;우선 미리 학습된 생성 모델을 통해 이미지 처리 모델을 구현하려면 한 가지 추가적인 핵심적인 기술을 필요로 합니다. 바로 GAN inversion이라 불리는 기술입니다. 기존의 생성 모델에서 generator $$G$$가 무작위로 생성된 latent code $$z$$를 입력으로 받아 가상의 이미지 $$G(z)$$를 생성해 출력했다면, GAN inversion은 실제 이미지 $$x$$와 특정 reconstruction loss $$\mathcal{L}(\cdot)$$에 대해 다음 조건을 만족하는 latent code $$z^*$$를 출력합니다. 

$$z^* = \arg\min_{z \in \mathcal{Z}} \mathcal{L}(G(z), x)$$

만약 $$\mathcal{L}(\cdot)$$을 단순한 pixel-wise loss라고 생각한다면 이 식을 푸는 것은 $$G$$에 입력됐을 때 $$x$$와 동일한 이미지가 나오는 latent code를 찾는 문제가 됩니다. 이름 그대로 이미지를 입력으로 받아 해당하는 latent code를 출력하는 GAN의 역변환이라고 볼 수 있을 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;여기서 주어진 문제가 어떤 종류의 이미지 처리에 대한 것인지에 따라 loss function을 다르게 정의할 수 있습니다. 예를 들어, 흑백 이미지를 컬러 이미지로 변환하는 image colorization 문제라면 다음과 같이 이미지의 휘도에 해당하는 채널(YCbCr 컬러 공간에서 Y 채널)만을 취해 흑백 이미지를 만드는 전처리 함수 $$\texttt{gray}(\cdot)$$를 써서 loss function을 새로 정의할 수 있습니다.

$$\mathcal{L}_{color} = \mathcal{L}(\texttt{gray}(G(z)), I_{gray})$$

이를 이용한 back-propagation으로 latent code $$z$$를 최적화하는 것은 흑백으로 만들었을 때 입력 흑백 이미지와 동일하게 되는 이미지를 미리 학습된 생성 모델의 latent space에서 찾는 것과 같을 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;이러한 방식으로 생성 모델을 이미지 처리에 활용하려는 연구가 여럿 있었지만, 앞서 설명했듯 좋은 성과를 내지는 못했습니다. 이 논문의 저자들은 그 이유를 기존 연구들이 단일 latent code만을 최적화하려고 했기 때문이라고 분석했습니다. 더 정확히 말하자면, 단일 latent code만을 다루게 되면 생성 모델의 latent space에 정답이 되는 이미지가 포함이 되지 않을 때 해당 이미지에 대한 완벽한 inversion이 존재하지 않게 되므로 정답을 아예 찾을 수 없는 것입니다.

&nbsp;&nbsp;&nbsp;&nbsp;이 논문은 여러 개의 latent code를 이용한 GA
N inversion 방법을 제시하면서 다음과 같은 contribution을 가지고 있다고 주장합니다.
- 여러 개의 latent code와 adaptive channel importance를 활용한 새로운 GAN inversion 방법인 mGANprior(multi-code GAN prior)를 제시함
- mGANprior를 image colorization, SISR, image inpainting 등 다양한 실제 이미지 처리 문제에 적용시킴
- GAN generator의 각 layer에 있는 feature들을 결합하여 각각 다른 layer에서의 internal representation을 분석하는 방법을 제시함

# Method

&nbsp;&nbsp;&nbsp;&nbsp;mGANprior는 그 이름에 걸맞게 여러 개의 latent code $$\{z_n\}^N_{n=1}$$를 사용합니다. 이 latent code들은 기본적으로 적절한 방식으로 합쳐져서 하나의 이미지를 표현할 수 있어야 합니다. 각각의 latent code들을 어떻게 정의해야 할까요?

&nbsp;&nbsp;&nbsp;&nbsp;단순히 생각하면, 다음과 같이 latent code들의 단순합에 대응되는 이미지는 각각의 latent code들에 대응되는 이미지의 단순합이 되도록 정의할 수 있을 것 같습니다.

$$G\left(\sum_{n=1}^N z_n\right) = \sum_{n=1}^N x_n$$

하지만 실제로는 생성 모델의 latent space는 선형 공간이 아니기 때문에 이와 같은 단순한 정의는 불가능합니다. 따라서 이들을 합치는 다른 방법을 생각해야 합니다.

&nbsp;&nbsp;&nbsp;&nbsp;최근 연구를 통해 GAN inversion에서 latent code 자체를 분석하는 것보다 generator의 중간 layer에서 feature를 분석하는 것이 더 효과적이라는 것이 알려졌습니다. 이를 채용하여 저자들은 입력으로 들어오는 latent code가 아니라 generator 네트워크에서의 intermediate feature map들을 합치는 방식을 채용하기로 했습니다. 

&nbsp;&nbsp;&nbsp;&nbsp;먼저 $$G(\cdot)$$을 두 개의 sub-network $$G_1^{(\ell)}(\cdot)$$과 $$G_2^{(\ell)}(\cdot)$$으로 나누어야 합니다. 이를 이용하여 generator의 $$\ell$$번째 layer의 feature map을 다음과 같이 정의할 수 있습니다.

$$F_n^{(\ell)} = G_1^{(\ell)}(z_n)$$

여기에 학습 가능한 parameter $$\alpha_n \in \mathbb{R}^C$$를 이용해 다음과 같은 weighted sum을 구할 수 있습니다.

$$\sum_{n=1}^{N} F_n^{(\ell)} \odot \alpha_n$$

이는 $$F_n^{(\ell)}$$의 각 채널 중 어떤 채널에 더 큰 비중을 둘지를 결정하는 adaptive channel importance를 적용해 각 feature map을 하나로 합치는 것으로 볼 수 있습니다. 이제 합쳐진 feature map을 generator의 나머지 부분인 $$G_2^{(\ell)}$$에 입력함으로써 결과 이미지를 생성할 수 있습니다.

$$x^{inv} = G_2^{(\ell)} \left(\sum_{n=1}^{N} F_n^{(\ell)} \odot \alpha_n\right)$$

&nbsp;&nbsp;&nbsp;&nbsp;이제 우리가 풀어야 할 문제를 다음과 같은 식으로 표현할 수 있습니다.

$$\{ z_n^* \}_{n=1}^N,\{ \alpha_n^* \}_{n=1}^N = \argmin_{\{ z_n \}_{n=1}^N,\{ \alpha_n \}_{n=1}^N} \mathcal{L}(x^{inv}, x)$$

저자들은 reconstruction loss $$\mathcal{L}$$을 다음과 같이 pixel-wise error와 perceptual error의 합으로 정의했습니다.

$$\mathcal{L}(x_1, x_2) = \lVert x_1-x_2 \rVert_2^2 + \lVert \phi(x_1) - \phi(x_2) \rVert_1$$

여기서 $$\phi(\cdot)$$는 미리 학습된 VGG-16에 이미지를 입력해서 얻을 수 있는 intermediate feature map을 의미합니다. 이렇게 되면 reconstruction loss는 두 이미지의 low-level에서의 거리와 high-level에서의 거리를 모두 반영하게 됩니다.

&nbsp;&nbsp;&nbsp;&nbsp;이제 이 방법을 실제 이미지 처리 task에 적용해야 합니다. 본 논문에서는 image colorization, SISR, 그리고 image inpainting까지 총 세 가지의 이미지 처리 task를 다루었습니다. 각각의 task는 다음과 같은 reconstruction loss를 최적화함으로써 수행됩니다.

$$\mathcal{L}_{color} = \mathcal{L} \left( \texttt{gray}(x^{inv}), I_{gray} \right)$$

$$\mathcal{L}_{SR} = \mathcal{L} \left( \texttt{down}(x^{inv}), I_{LR} \right)$$

$$\mathcal{L}_{inp} = \mathcal{L} \left( x^{inv} \circ m, I_{ori} \circ m \right)$$

차례대로 $$\texttt{gray}(\cdot)$$은 휘도 채널 추출, $$\texttt{down}(\cdot)$$은 downscaling, 그리고 $$m$$은 가려진 픽셀 위치를 0, 나머지를 1로 채운 binary mask를 의미하고 $$I_{gray}, I_{LR}, I_{ori}$$는 입력 이미지를 가리킵니다.

# Experiments

&nbsp;&nbsp;&nbsp;&nbsp;본 논문에서는 PGGAN과 StyleGAN을 이용해 mGANprior를 구현하였으며, 얼굴을 위해 CelebA-HQ와 FFHQ 데이터셋이, 교회 및 침실 등의 사진을 위해서 LSUN 데이터셋이 사용되었습니다.

&nbsp;&nbsp;&nbsp;&nbsp;mGANprior를 구현하는 데 있어서 가장 중요한 요소는 바로 몇 개의 latent code를 사용할지입니다. 저자들은 실험을 통해 더 많은 latent code를 쓸 수록 계산량이 많아진다는 trade-off가 있지만, 그 수가 무한히 늘어난다고 해서 성능이 무한히 좋아지지는 않는다는 것을 보였습니다. 그들이 찾은 최적의 개수는 20개였고, 모든 실험은 이를 기준으로 진행되었습니다.

&nbsp;&nbsp;&nbsp;&nbsp;또한 generator에서 intermediate feature를 합칠 지점을 정하는 것도 중요한 사항입니다. 8개의 layer를 가진 PGGAN으로 실험한 결과, 더 뒤에 있는 layer를 선택할수록 더 좋은 결과가 나온다는 것이 확인되었습니다. 이는 뒷단에 있는 layer일수록 전체적인 semantics보다는 이미지의 패턴이나 테두리, 색과 같은 디테일한 부분들에 대한 정보를 담고 있기 때문에 더 좋은 품질의 결과를 낸다고 해석할 수 있습니다.

&nbsp;&nbsp;&nbsp;&nbsp;
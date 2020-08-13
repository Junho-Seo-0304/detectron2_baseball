# detectron2_baseball
인턴 기간동안 공부한 Detectron2

테스트에 이용된 알고리즘은 두 가지로 Retinanet과 Faster R-CNN입니다.

- R-CNN

 R-CNN이란 Region with convolutional neural network로 이는 이미지나 영상에 Selective search 즉, 선택적 탐색을 이용하여, 객체가 있을 법한 영역 제안(Region Proposal)을 찾고 각 영역 제안에 컨볼루션 신경망을 적용하여 객체를 분류한 뒤, 객체의 위치를 보정하는 방법입니다. 이 방법은 각 영역 제안마다 적용되어야 하므로 전체적으로 시간이 오래 걸립니다. 이것을 해결하기 위해 Fast R-CNN과 Faster R-CNN이란 방법이 나왔습니다. Fast R-CNN은 기존 R-CNN 방식에 ROI Pooling(Region of interest pooling)이라는 layer를 도입하여 컨볼루션 신경망에서 얻어진 Feature map의 일부 영역으로부터 정규화된 특징을 추출하는 방법입니다. 즉, 각 영역 제안에 CNN을 반복 적용하지 않고 입력된 데이터에 CNN은 한 번만 적용하고 ROI Pooling으로 객체를 판별하기 위한 특징을 뽑아내 시간을 대폭 줄였습니다. 더 나아가 Faster R-CNN은 선택적 탐색으로 영역 제안을 추출하던 기존 방법을 딥러닝 방법을 대체하여 속도를 개선한 알고리즘입니다. 

https://brunch.co.kr/@kakao-it/66

 ROI Pooling은 CNN의 output으로 나온 각각의 크기가 다른 Feature maps를 같은 크기로 변환하여 특징을 찾을 수 있게 해주는 기법입니다. 그래서 Faster R-CNN을 이용하면 크기가 다른 물체도 detection 할 수 있게 됩니다.

http://incredible.ai/deep-learning/2018/03/17/Faster-R-CNN/#:~:text=Region%20of%20Interest%20Pooling,-RPN%EC%9D%B4%ED%9B%84%2C%20%EC%84%9C%EB%A1%9C&text=%EC%84%9C%EB%A1%9C%20%EB%8B%A4%EB%A5%B8%20%ED%81%AC%EA%B8%B0%EB%9D%BC%EB%8A%94%20%EB%A7%90,%EB%8B%A4%EB%A5%B8%20%ED%81%AC%EA%B8%B0%EB%9D%BC%EB%8A%94%20%EB%9C%BB%EC%9E%85%EB%8B%88%EB%8B%A4.&text=%EC%9D%B4%EB%95%8C%20%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94%20%EA%B8%B0%EB%B2%95%EC%9D%B4,Pooling(ROI)%20%EA%B8%B0%EB%B2%95%EC%9E%85%EB%8B%88%EB%8B%A4.

- Retinanet

 Retinanet은 정해진 위치와 정해진 크기의 객체만 찾는 알고리즘입니다. 이 알고리즘은 보통 원본 이미지를 고정된 사이즈 그리드 영역으로 나누어 각 영역에 대해 형태와 크기가 미리 결정된 객체의 고정 개수를 예측합니다.

https://blogsaskorea.com/156

 detect 알고리즘에는 1-stage detector와 2-stage detector로 나뉩니다. 2-stage detector는 Regional Proposal 즉, 영역 제안과 Classification이 순차적으로 이루어집니다. 이것은 classification과 localization 문제를 순차적으로 해결한다는 뜻입니다. 하지만 1-stage detector는 2-stage detector와 반대로 regional proposal과 classification이 동시에 이루어집니다. 그래서 2-stage detector보다 빠르지만, 정확도가 낮습니다. 여기서 retinanet은 1-stage detector에 속하고, faster R-CNN은 2-stage detector에 속합니다. 

https://ganghee-lee.tistory.com/34

 setup 구분하는 테스트 결과 retinanet은 확실히 추정하는 것에 퍼센트가 낮았으며 faster R-CNN은 IOU가 평균 98%의 정확도를 보여주었습니다. 하지만 이 detector 알고리즘들은 pose estimation 즉, 포즈를 판단하는 알고리즘이 아니라 사물을 detection 하는 Object Detection 알고리즘이라 사람 3명이 같이 있는 모습은 setup으로 판단하는 오탐이 발생하게 되었습니다. pose estimation을 할 수 있으려면 Object Detection이 아닌 Instance Segmentation을 이용해야 합니다. 알고리즘은 두 가지 알고리즘이 있는데 Keypoint R-CNN과 Mask R-CNN이 있습니다.

- Keypoint Detection

 Keypoint R-CNN은 말 그대로 사람의 자세에서 Keypoint를 잡아 자세의 뼈대를 만들어 자세에 대한 detection을 해주는 것입니다. JSON 파일을 만들 때 지금까지의 데이터셋은 annotation에서 bbox만 입력을 하였다면 Keypoint detection은 keypoint와 이 keypoint를 잇는 skeleton을 선언해주어야 합니다. 여기서 Keypoint들은 x 좌표, y 좌표 그리고 v 값으로 정의가 되는데 v 값은 현재 그 포인트가 사물에 가려져서 안보이는지 보이는지의 값입니다. v=0일 때는 분류가 되지 않음, v=1일 때 분류될 값이지만 보이지 않음, v=2일 때 분류될 값이며 보이는 상태입니다. skeleton은 이 포인트들을 이어주는 edge를 선언하는 부분입니다.

https://cocodataset.org/#format-data

- Mask R-CNN

 Mask R-CNN은 Faster R-CNN에 각 픽셀이 객체인지 아닌지를 masking 하는 CNN을 추가한 R-CNN입니다. Mask R-CNN은 instance seg, bounding-box, object detection, person keypoint detection에서 이전 모델보다 우수한 성능을 보인다고 합니다.

 Mask R-CNN과 Faster R-CNN의 큰 차이점은 3가지가 있습니다.
1. Faster R-CNN에 존재하는 bbox 인식에 병렬로 object mask 예측을 추가
 기존의 Object Detection 역할은 Faster R-CNN이 하고 각각의 ROI(Region of Interest)에 Mask segmentation을 해주는 FCN(Fully Connected Network)을 추가하였습니다.
2. ROI Pooling 대신 ROI Align을 사용
 기존의 R-CNN은 object detection을 위한 모델이었기 때문에 ROI Pooling에서 정확한 위치 정보를 담는 것이 중요하지 않았지만, Mask R-CNN은 pixel-by-pixel로 하는 detection이기 때문에 위치 정보가 필요합니다. 그래서 ROI Align 방식을 이용하여 위치 정보를 포함된 detection 할 물체의 특징을 찾을 수 있게 합니다. 여기서 그리드 포인트로부터 샘플링 포인트의 값을 계산하는 방식이 bilinear interpolation이라고 합니다.
3. Mask prediction과 class prediction을 분리
 Mask prediction과 class prediction을 분리하여 mask prediction에서 다른 클래스를 고려할 필요 없이 binary mask를 predict 하면 되기 때문에 성능의 향상을 보입니다.

https://mylifemystudy.tistory.com/82

 위의 설명 말고도 detectron2에서 이용 가능한 방법이 몇 가지가 더 있습니다. 하지만 야구 중계 자동 하이라이트 생성에선 쓰이지 않으리라고 보여 간단히 설명하겠습니다. 두 가지가 있는데 Panoptic Segmentation과 cityscape가 있습니다. Panoptic은 배경을 stuff로, 배경을 제외한 객체를 thing으로 나누어 detection 하는 방법이며, cityscape는 독일의 도시들을 위주로 한 도시 배경에 대한 Dataset 을 제공하여 flat, human, vehicle, construction, object, nature, sky 등을 semantic segmatation을 해주는 방법입니다.

https://89douner.tistory.com/113

 따라서 bbox만 이용하는 현 상황상 detectron2에서 이용하는 COCO challenge들은 부적합하다고 판단됩니다. COCO challenge를 이용하려면 최소한 keypoint 값을 이용하여 사람의 자세를 판단할 수 있어야 합니다.

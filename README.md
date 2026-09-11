# MNIST 손글씨 숫자 인식 패드

손가락, 펜, 마우스로 숫자 하나를 쓰면 브라우저에서 예측합니다.
학습된 모델을 포함하므로 사이트 실행을 위해 Python 설치나 재학습이 필요하지 않습니다.

## GitHub Pages에 게시하기

1. GitHub에서 `mnist-pad`라는 저장소를 만듭니다. 무료 계정이라면 Public을 선택하세요. 코드와 사이트가 공개됩니다.
2. 이 폴더 **안의 파일들**을 저장소 최상위에 올립니다. 저장소를 열었을 때 `index.html`, `app.js`, `model.bin` 등이 바로 보여야 합니다. ZIP 자체를 업로드하지 마세요.
3. 저장소의 Settings → Pages로 이동합니다.
4. Source에서 Deploy from a branch를 선택합니다.
5. Branch는 main, 폴더는 /(root)를 선택하고 Save를 누릅니다.
6. 게시가 끝나면 같은 화면의 Visit site로 접속합니다. 변경 사항이 반영되는 데 최대 10분 걸릴 수 있습니다.

일반적인 주소 형식: `https://사용자명.github.io/mnist-pad/`.

모든 앱 내부 경로는 상대 경로이므로 저장소 이름을 바꿔도 동작합니다.
`.nojekyll`은 Jekyll 빌드를 생략하는 파일입니다. 파일 탐색기에서 숨겨져 있을 수 있습니다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| index.html | 화면 구성 |
| style.css | 디자인과 반응형 레이아웃 |
| app.js | 필기 입력, 전처리, 결과 표시 |
| inference.mjs | 모델 순전파 및 softmax 계산 |
| model.bin | 학습된 float32 가중치 |
| model.json | 모델 구조와 평가 정보 |
| train_model.py | 로컬 재학습 코드 |
| requirements.txt | 재학습용 Python 라이브러리 |

## 모델과 평가

MNIST 원본 학습 데이터 60,000장 중 54,000장을 학습, 6,000장을 검증에 사용했습니다.
MLP 구조는 784 → 128(ReLU) → 64(ReLU) → 10(softmax)입니다.
검증 성능으로 조기 종료한 모델의 별도 테스트 10,000장 정확도는 97.70%입니다.
이는 MNIST 테스트 정확도이며, 직접 패드에 쓴 숫자의 정확도를 보장하지 않습니다.
숫자별 softmax 확률 역시 정답 확률로 보정된 값은 아닙니다.

전처리: 필기 영역 잘라내기 → 긴 변 20픽셀로 축소 → 28×28에 배치 → 무게중심 정렬 → 픽셀값을 255로 나누기.

## 로컬 실행 및 재학습

사이트 실행: 이 폴더에서 `python -m http.server 8000` 실행 후 브라우저로 `http://localhost:8000` 접속.
HTML 파일을 더블 클릭하는 방식은 브라우저의 모듈/파일 요청 제한 때문에 적합하지 않습니다.

재학습이 필요한 경우에만:

```bash
pip install -r requirements.txt
python train_model.py /path/to/mnist.npz
```

데이터 다운로드: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

GitHub Pages는 Python 학습 코드를 실행하지 않습니다. 학습은 로컬에서 하고, 사이트는 저장된 모델을 JavaScript로 실행합니다.

## 검증 범위

원본 모델과 JavaScript 추론의 수치 일치 확인을 완료했습니다.
GitHub Pages용 상대 경로와 로컬 파일 참조를 확인했습니다.
GitHub 계정 연결 전이므로 실제 GitHub Pages 게시 및 브라우저 테스트는 아직 수행하지 않았습니다.

## 공식 안내

- https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
- https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

# 영화 평점 예측

## About
DavianLab 연구실에서 딥러닝 스터디 및 pytorch 스터디에서 진행하기 위해 만든 코드입니다.

jupyter notebook에서 작동이 잘 되는 것을 확인했고 이를 python 코드로 옮겼습니다.

저작권 등의 문제로 데이터는 올리지 않았습니다.

코드에 에러가 있거나 기타 문의는 rudvlf0413@korea.ac.kr로 메일 주세요!


## Dataset
	네이버 영화(movie.naver.com)
	Input: 영화 리뷰, Target: 영화 평점


## Model
* CNN을 이용한 Rigression (model.py 참조)
* Multichannel Variable-size Convolution


## Dependency
	Python 3.6
	Linux(Ubuntu) or MacOS
	PyTorch 0.1.12, GPU 사용
	Konlpy(Twitter 모듈)


## Result Examples
* 이 영화 겁나 재미있다 ㅋㅋㅋㅋ 겁나 웃겨

	prediction score: 0.932316

* 뭐야 기대하고 봤는데 별로네

	prediction score: 0.398688
	
* 그럭저럭

	prediction score: 0.587084
	
* 영화가 왜 이렇게 감동적이야 ㅠㅠㅠㅠ

	prediction score: 0.749872
	
* 이 영화를 볼 바에 기부를 하겠다

	prediction score: 0.330613
	
* 팝콘 소리 때문에 집중이 안됐다 ㅡㅡ

	prediction score: 0.410676
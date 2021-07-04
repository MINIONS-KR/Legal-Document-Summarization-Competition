# Legal-Document-Summarization-Competition
2021 AI Online Competition (TEAM Minions)

### Code file의 상세설명

```bash
.
├── ARA
│   ├── config
│   │   ├── predict_config.yml
│   │   └── train_config.yml
│   ├── data
│   │   ├── kobert-0-train.pkl
│   │   ├── kobert-0-valid.pkl
│   │   ├── kobert-1-train.pkl
│   │   ├── kobert-1-valid.pkl
│   │   ├── kobert-2-train.pkl
│   │   ├── kobert-2-valid.pkl
│   │   ├── kobert-3-train.pkl
│   │   ├── kobert-3-valid.pkl
│   │   ├── kobert-4-train.pkl
│   │   └── kobert-4-valid.pkl
│   ├── data_split.py
│   ├── kfold_train.py
│   ├── model
│   │   └── model.py
│   ├── modules
│   │   ├── dataset.py
│   │   ├── earlystoppers.py
│   │   ├── metrics.py
│   │   ├── recorders.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── predict.py
│   └── train.py
├── IKHYO
│   ├── configs
│   │   ├── inference
│   │   │   └── base_config.json
│   │   └── train
│   │       └── base_config.json
│   ├── model
│   │   └── model.py
│   ├── modules
│   │   ├── criterion.py
│   │   ├── dataset.py
│   │   ├── earlystoppers.py
│   │   ├── metrics.py
│   │   ├── optimizer.py
│   │   ├── recorders.py
│   │   ├── scheduler.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── predict.py
│   └── train.py
├── MINYONG
│   ├── configs
│   │   ├── inference
│   │   │   └── base_config.json
│   │   └── train
│   │       └── base_config.json
│   ├── model
│   │   └── model.py
│   ├── modules
│   │   ├── criterion.py
│   │   ├── dataset.py
│   │   ├── earlystoppers.py
│   │   ├── metrics.py
│   │   ├── optimizer.py
│   │   ├── recorders.py
│   │   ├── scheduler.py
│   │   └── utils.py
│   ├── predict.py
│   └── train.py
├── models
│   ├── bertsum0.pt
│   ├── bertsum1.pt
│   ├── bertsum2.pt
│   ├── bertsum3.pt
│   ├── bertsum4.pt
│   ├── Ik_fold0.pt
│   ├── Ik_fold1.pt
│   ├── Ik_fold2.pt
│   ├── Ik_fold3.pt
│   ├── Ik_fold4.pt
│   ├── kobert0.pt
│   ├── kobert1.pt
│   ├── kobert2.pt
│   ├── kobert3.pt
│   ├── kobert4.pt
│   ├── koelectra.pt
│   └── sentavg.pt
├── README.md
├── requirements.txt
└── train.sh
```

**`ARA` 폴더** : KoBert, KoElectra, SentAvg(custom) 모델 학습 및 추론
- `config`: 학습과 추론시에 설정할 환경들을 저장하는 yml config file이 존재하는 폴더
  - `predict_config.yml`: 추론시에 사용할 필요한 환경 전달
  - `train_config.yml`: 학습시에 사용할 하이퍼 파라미터 및 필요한 환경 전달
- `data`: 5 Fold에 사용되는 데이터를 저장하는 폴더
  - `kobert-0-train.pkl ~ kobert-4-train.pkl`: KoBert의 해당 Fold에서 사용할 train 파일
  - `kobert-0-valid.pkl ~ kobert-4-valid.pkl`: KoBert의 해당 Fold에서 사용할 validation 파일
- `model > model.py`: KoBert, KoElectra, SentAvg 모델 구현 파일(pytorch)
- `modules`
    - `dataset.py`: 학습에 사용될 pytorch dataset 정의 파일
    - `earlystoppers.py`: 학습시에 Overfitting을 방지하기 위해 Hitrate Score 또는 loss를 기준으로 EarlyStopping을 할 수 있는 객체 정의
    - `metrics.py`: Hitrate Score를 계산하는 함수 정의
    - `recorders.py`: 모델 저장 작업 및 logging 작업을 하는 PerformanceRecorder 객체 정의
    - `trainer.py`: 모델 training, validation, test 작업을 하는 Trainer 객체 정의
    - `utils.py`: Config File Parsing, Seed 통일 등 학습과 추론에 필요한 다양한 util 함수 정의
- `data_split.py`: 5 Fold로 훈련하는데 사용할 데이터를 나누어 저장하는 파일
- `kfold_train.py`: 훈련 데이터를 5 Fold로 나누어서 KoBert를 학습하는 파일
- `predict.py`: 학습된 모델에 대해서 prediction을 하는 파일 (model_name argument 필요)
- `train.py`: KoElectra와 SentAvg를 학습하는 파일 (model_name argument 필요)

<br>

**`IKHYO` 폴더** : OneSentenceBert(custom) 모델 학습 및 추론
- `config`: 학습과 추론시에 설정할 환경들을 저장하는 json config file이 존재하는 폴더
  - `train > base_config.json`: 학습시에 사용할 하이퍼 파라미터 및 필요한 환경 전달
  - `inference > base_config.json`: 추론시에 사용할 필요한 환경 전달
- `model > model.py`: OneSentenceBert(custom) 모델 구현 파일(pytorch)
- `modules`
    - `criterion.py`: loss function을 정의하는 file로 config file을 통해 지정한 대상을 학습시에 사용
    - `dataset.py`: OneSentenceBert(custom) 모델에 사용될 pytorch dataset 정의 파일
    - `earlystoppers.py`: 학습시에 Overfitting을 방지하기 위해 Hitrate Score 또는 loss를 기준으로 EarlyStopping을 할 수 있는 객체 정의
    - `metrics.py`: Hitrate Score를 계산하는 함수 정의
    - `optimizer.py`: optimizer를 정의하는 file로 config file을 통해 지정한 대상을 학습시에 사용
    - `recorders.py`: 모델 저장 작업 및 logging 작업을 하는 PerformanceRecorder 객체 정의
    - `scheduler.py`: scheduler를 정의하는 file로 config file을 통해 지정한 대상을 학습시에 사용
    - `utils.py`: Config File Parsing, Seed 통일 등 학습과 추론에 필요한 다양한 util 함수 정의
- `results`: train시에 logging 기록들을 저장하는 공간
- `predict.py`: 5 Fold로 학습된 각각의 모델에 대해서 prediction을 하는 파일
- `train.py`: 훈련 데이터를 5 Fold로 나누어서 OneSentenceBert(custom) model을 학습하는 파일

<br>

**`MINYONG` 폴더** : BertSumExt 모델 학습 및 추론
- `config`: 학습과 추론시에 설정할 환경들을 저장하는 json config file이 존재하는 폴더
  - `train > base_config.json`: 학습시에 사용할 하이퍼 파라미터 및 필요한 환경 전달
  - `inference > base_config.json`: 추론시에 사용할 필요한 환경 전달
- `model > model.py`: BertSumExt 모델 구현 파일(pytorch)
- `modules`
    - `criterion.py`: loss function을 정의하는 file로 config file을 통해 지정한 대상을 학습시에 사용
    - `dataset.py`: BertSumExt 모델에 사용될 pytorch dataset 정의 파일
    - `earlystoppers.py`: 학습시에 Overfitting을 방지하기 위해 Hitrate Score 또는 loss를 기준으로 EarlyStopping을 할 수 있는 객체 정의
    - `metrics.py`: Hitrate Score를 계산하는 함수 정의
    - `optimizer.py`: optimizer를 정의하는 file로 config file을 통해 지정한 대상을 학습시에 사용
    - `recorders.py`: 모델 저장 작업 및 logging 작업을 하는 PerformanceRecorder 객체 정의
    - `scheduler.py`: scheduler를 정의하는 file로 config file을 통해 지정한 대상을 학습시에 사용
    - `utils.py`: Config File Parsing, Seed 통일 등 학습과 추론에 필요한 다양한 util 함수 정의
- `results`: train시에 logging 기록들을 저장하는 공간(loss graph, score graph, log in CSV, 학습시 사용한 config file, 학습시 logger를 통해 기록된 모든 log)
- `predict.py`: 5 Fold로 학습된 각각의 모델에 대해서 prediction을 하는 파일
- `train.py`: 훈련 데이터를 5 Fold로 나누어서 BertSumExt를 학습하는 파일

<br>

**`models` 폴더** : 학습된 모든 모델을 저장하는 공간
- `Ik_fold0 ~ Ik_fold4.pt` : 5-Fold를 기준으로 학습된 OneSentenceBert 모델
- `bertsum0 ~ bertsum4.pt` : 5-Fold를 기준으로 학습된 BertSumExt 모델
- `kobert0 ~ kobert4.pt` : 5-Fold를 기준으로 학습된 KoBert Base 모델
- `sentavg.pt` : 학습된 SentAvg 모델
- `koelectra.pt` : 학습된 KoElectra Base 모델

<br>

### output에 대한 description

```bash
.
├── submissions
│   ├── bertsum0.json
│   ├── bertsum1.json
│   ├── bertsum2.json
│   ├── bertsum3.json
│   ├── bertsum4.json
│   ├── final_submission.json
│   ├── ikhyo0.json
│   ├── ikhyo1.json
│   ├── ikhyo2.json
│   ├── ikhyo3.json
│   ├── ikhyo4.json
│   ├── ikhyo4.json
│   ├── bertsum0.json
│   ├── bertsum1.json
│   ├── bertsum2.json
│   ├── bertsum3.json
│   ├── bertsum4.json
│   ├── kobert0.json
│   ├── kobert1.json
│   ├── kobert2.json
│   ├── kobert3.json
│   ├── kobert4.json
│   ├── koelectra.json
│   ├── sample_submission.json
│   └── sentavg.json
├── ensemble.py
└── inference.sh
```

**`submissions` 폴더**: 제출에 필요한 모든 prediction json file을 저장하는 공간
- `final_submission.json`: 앙상블을 통한 최종 prediction file
- `bertsum0 ~ bertsum4.json` : BertSumExt 모델의 5-Fold Result
- `ikhyo0 ~ ikhyo4.json` : OneSentenceBert 모델의 5-Fold Result
- `kobert0 ~ kobert4.json` : KoBert Base 모델의 5-Fold Result
- `sentavg.json` : SentAvg 모델의 Result
- `koelectra.json` : KoElectra Baseline 모델의 Result
- `sample_submission.json`: 결과물을 생성하기위해 존재하는 prediction form

<br>

**`ensemble.py`**
- submissions 폴더 내에 있는 json 파일로(sample_submission.json 제외) hard voting 기반 ensemble을 수행하여 최종 제출 파일 생성

<br>

### 학습에 필요한 명령어

(Minions Folder 내부에서) `bash train.sh`

### 추론에 필요한 명령어

(Minions Folder 내부에서) `bash inference.sh`

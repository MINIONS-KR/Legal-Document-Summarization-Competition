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

**`ARA` 폴더**
- 각 폴더에 대한 설명 작성하기

**`IKHYO` 폴더**
- 각 폴더에 대한 설명 작성하기

**`MINYONG`폴더**
- 각 폴더에 대한 설명 작성하기

**`models`폴더**
- 각 폴더에 대한 설명 작성하기

### output에 대한 description

```bash
.
├── submissions
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

**`submissions`폴더**
- 각 폴더에 대한 설명 작성하기

**`ensemble.py`**
- ensemble.py에 대한 설명 작성하기


### 학습에 필요한 명령어


### 추론에 필요한 명령어

## Restore punct and capu
#### Model Architecture:
  + EnViBERT (cased) + BiLSTM + FFW
  + EnViBERT (uncased) + BiLSTM + FFW
  + XLMR + BiLSTM + FFW

#### Library:
  + pytorch
  + transformers
  + tqdm
  + numpy
  + sklearn
  
#### Data : 
  + 28G dữ liệu báo
  + 1G subfilm
  + 50MB dữ liệu chatbot
  
## Kết quả thử nghiệm:  
#### EnViBERT (cased) + BiLSTM + FFW:
Label | precision | recall | F1-score
---|---|---|---
`O` | 0.98 | 0.98 | 0.98 
`O.` | 0.90 | 0.92 | 0.91 
`O,` | 0.80 | 0.75 | 0.78 
`UPPER` | 0.94 | 0.94 | 0.94
`UPPER.` | 0.90 | 0.92 | 0.91
`UPPER,` | 0.86 | 0.78 | 0.82

#### XLMR + BiLSTM + FFW
Label | precision | recall | F1-score
---|---|---|---
`O` | 0.98 | 0.98 | 0.98 
`O.` | 0.92 | 0.93 | 0.93 
`O,` | 0.82 | 0.76 | 0.79 
`UPPER` | 0.94 | 0.95 | 0.95
`UPPER.` | 0.91 | 0.94 | 0.92
`UPPER,` | 0.88 | 0.80 | 0.83
# general
bert="envibert_cased"
# bert="envibert_uncased"
# bert="xlmr"
mode="train"

nb_labels= 9
hidden_dim_lstm=256
drop_rate=0.1
random_seed=8888
weight_decay=5e-6
lr=6e-5
max_epoch=8
max_sent_length_option=64
max_sent_length=256
freeze=True
num_freeze_layers=3
yield_every=256
num_cores=40

train_bs=128
val_bs=128
test_bs=128

parallel=False
cuda="cuda:0"
device_ids=[0,1,2,3]

# path
train_path="/home/tuyendv/projects/text-restoration-tag-label/temp/test"
test_path="/home/tuyendv/datas/datas/chatbot/test"
val_path="/home/tuyendv/projects/text-restoration-tag-label/temp/test"

tag2index="src/resources/dict/tag2index.json"
index2tag="src/resources/dict/index2tag.json"

checkpoint_path=f'checkpoint/{bert}/checkpoint_{bert}_%EPOCH%.pt'
res_path=f"results/f1-score/{bert}/acc_{bert}_%EPOCH%.txt"
confusion_matrix_path=f"results/confusion-matrix/{bert}/confusion_matrix_{bert}_%EPOCH%.jpg"

warm_up="checkpoint/envibert_cased/checkpoint_envibert_cased_1_news.pt"
test_checkpoint="checkpoint/checkpoint_envibert_cased_1-removed-module.pt"

pretrained_envibert_uncased ='src/resources/pretrained/uncased'
pretrained_envibert_cased= 'src/resources/pretrained/envibert'

if bert=="envibert_uncased":
    hidden_dim_bert=512
    toknizer_path="src/resources/pretrained/uncased"
elif bert=="envibert_cased":
    hidden_dim_bert=768
    toknizer_path="src/resources/pretrained/envibert"
elif bert=="xlmr":
    hidden_dim_bert=768
    toknizer_path=None
    
# prob
prob_cutting_both_sides=0.2
prob_cutting_left_sides=0.2
prob_cutting_right_sides=0.2
prob_stay_same=0.4

prob_list=["cut_both_sides", "cut_left_sides", "cut_right_sides", "stay_same"]
prob_weight=[prob_cutting_both_sides, prob_cutting_left_sides, prob_cutting_right_sides, prob_stay_same]

# label
PAD_TOKEN, PAD_TOKEN_ID="<pad>", 1
BOS_TOKEN, BOS_TOKEN_ID="<s>", 0
EOS_TOKEN, EOS_TOKEN_ID="</s>", 2

PAD_TAG, PAD_TAG_ID="<pad>", 1
BOS_TAG, BOS_TAG_ID="<s>", 0
EOS_TAG, EOS_TAG_ID="</s>", 2
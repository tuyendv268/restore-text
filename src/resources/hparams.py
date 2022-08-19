train_bs=8
val_bs=128
test_bs = 128

nb_labels= 11
embedding_dim = 768
hidden_dim=256
drop_rate = 0.1

weight_decay = 5e-6
lr = 6e-5
max_epoch = 10
max_sent_length = 256

yield_every = 256
num_cores = 3

mode = "train"
parallel = True
val_cuda=0
device_ids = [0,1,2,3]

bert = "envibert"
# bert = "envibert_uncased"
# bert = "xlmr"

prob_cutting_both_sides = 0.3
prob_cutting_left_sides = 0.2
prob_cutting_right_sides = 0.2
prob_stay_same = 0.3

prob_list = ["cut_both_sides", "cut_left_sides", "cut_right_sides", "stay_same"]
prob_weight = [prob_cutting_both_sides, prob_cutting_left_sides, prob_cutting_right_sides, prob_stay_same]


tag2index="src/resources/dict/tag2index.json"
index2tag="src/resources/dict/index2tag.json"

pretrained_uncased_envibert ='src/resources/pretrained/uncased'
pretrained_envibert= 'src/resources/pretrained/envibert'

checkpoint_path = f'checkpoint/{bert}/checkpoint_{bert}_%EPOCH%.pt'
res_path = f"results/f1-score/{bert}/acc_{bert}_%EPOCH%.txt"
confusion_matrix_path = f"results/confusion-matrix/{bert}/confusion_matrix_{bert}_%EPOCH%.jpg"

warm_up = "checkpoint/checkpoint_.pt"
test_checkpoint = "checkpoint/envibert/checkpoint_envibert_0.pt"

# /home/tuyendv/projects/tag-label-restore-punct/output_data/test
train_path = "data/train"
test_path = "data/train"
val_path = "data/train"


PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2

PAD_TAG, PAD_TAG_ID = "<pad>", 1
BOS_TAG, BOS_TAG_ID = "<s>", 0
EOS_TAG, EOS_TAG_ID = "</s>", 2

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="$THIS_DIR/../data"
SEED_N=1318

git clone https://github.com/HSLCY/GlossBERT
cd GlossBERT

python run_classifier_WSD_sent.py \
--task_name WSD \
--train_data_dir "$BASE_DIR/glossbert/train_gloss_ws.csv" \
--eval_data_dir "$BASE_DIR/glossbert/dev_gloss_ws.csv" \
--output_dir "$BASE_DIR/glossbert/output_gloss_ws/$SEED_N" \
--bert_model bert-base-uncased \
--do_train \
--do_eval \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 32 \
--eval_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--seed $SEED_N

python run_classifier_WSD_sent.py \
--task_name WSD \
--eval_data_dir "$BASE_DIR/glossbert/test_gloss_ws.csv" \
--output_dir "$BASE_DIR/glossbert/output_gloss_ws/$SEED_N" \
--bert_model "$BASE_DIR/glossbert/output_gloss_ws/$SEED_N" \
--do_test \
--do_lower_case \
--max_seq_length 512 \
--eval_batch_size 8 \
--seed $SEED_N

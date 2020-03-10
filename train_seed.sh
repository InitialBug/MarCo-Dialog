


# parser.add_argument('--seed', type=int, default=913, help="random seed for initialization")
# parser.add_argument('--log', type=str, default='log', help="log file")

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python train_generator.py --option train --model model/c4ad96d_seed$i --batch_size 384 --max_seq_length 50\
    --seed $i &
done


python main.py  --gpu 1 \
                --max_epoch 100 \
                --energy \
                --energy_loss \
                --open_loss \
                --lr 0.0001 \
                --energy_method sum \
                --pixel_wise \
                --distance pixel_sim \
                --dataset MiniImageNet \
                --shot 5 \
                --ahead_combine \
                --top_k 10 \
                --new_benchmark all \
                --query 5 \
                --eval_query 5 \
                --method Ours \
                --test \
                --test_model_path checkpoints/SFCM/20231203_212404


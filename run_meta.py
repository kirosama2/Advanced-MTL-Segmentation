

""" Generate commands for meta-train phase. """
import os

def run_exp(num_batch=50, shot=3, teshot=1, query=1, lr1=0.0005, lr2=0.005, base_lr=0.01, update_step=20, gamma=0.5):
    max_epoch = 200
    step_size = 20
    way=2 #Backround as a class included. Adjust accordingly.
    gpu=1
       
    the_command = 'python3 main.py' \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --train_query=' + str(quer


""" Generate commands for meta-train phase. """
import os

def run_exp(num_batch=50, shot=3, teshot=1, query=1, lr1=0.0005, lr2=0.005, base_lr=0.01, update_step=20, gamma
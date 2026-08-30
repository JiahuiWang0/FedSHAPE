from argparse import Namespace

best_args = {
    'fl_digits': {
        # === 基础 FedAvg 系列 ===
        'fedavg': {'local_lr': 0.001, 'local_batch_size': 64},
        'fedavgspo': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},
        'fedavgheal': {'local_lr': 0.001, 'local_batch_size': 64},
        'fedavgshape': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},

        # === FedProx 系列 ===
        'fedprox': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01},
        'fedproxspo': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01, 'spo_alpha': 0.05},
        'fedproxheal': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01},
        'fedproxshape': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01, 'spo_alpha': 0.05},

        # === Scaffold 系列 ===
        'scaffold': {'local_lr': 0.001, 'local_batch_size': 64},
        'scaffoldspo': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},
        'scaffoldheal': {'local_lr': 0.001, 'local_batch_size': 64},
        'scaffoldshape': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},

        # === MOON 系列 ===
        'moon': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5},
        'moonspo': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},
        'moonheal': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5},
        'moonshape': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},

        # === FedDyn 系列 ===
        'feddyn': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01},
        'feddynspo': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},
        'feddynheal': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01},
        'feddynshape': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},

        # === AFL & q-FFL (原版) ===
        'afl': {'local_lr': 0.001, 'local_batch_size': 64, 'afl_lr': 0.01},
        'qffl': {'local_lr': 0.001, 'local_batch_size': 64, 'q_param': 0.1},

        # === AFL/QFFL 组合方法 ===
        'fedavgafl': {'local_lr': 0.001, 'local_batch_size': 64, 'afl_lr': 0.01},
        'fedavgqffl': {'local_lr': 0.001, 'local_batch_size': 64, 'q_param': 0.1},

        'fedproxafl': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01, 'afl_lr': 0.01},
        'fedproxqffl': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01, 'q_param': 0.1},

        'moonafl': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5, 'afl_lr': 0.01},
        'moonqffl': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5, 'q_param': 0.1},

        'feddynafl': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01, 'afl_lr': 0.01},
        'feddynqffl': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01, 'q_param': 0.1},

        # ==========================================
        # 析因消融实验 (Factorial Ablation) 完整列表
        # ==========================================
        'fedsam': {'local_lr': 0.001, 'local_batch_size': 64, 'sam_rho': 0.05},
        'fedsamheal': {'local_lr': 0.001, 'local_batch_size': 64, 'sam_rho': 0.05},
        'spoheal': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},

        # FedAvg 系列消融
        'fedavgema': {'local_lr': 0.001, 'local_batch_size': 64},
        'fedavgdiv': {'local_lr': 0.001, 'local_batch_size': 64},
        'fedavgspoema': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},
        'fedavgspodiv': {'local_lr': 0.001, 'local_batch_size': 64, 'spo_alpha': 0.05},

        # FedProx 系列消融
        'fedproxema': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01},
        'fedproxdiv': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01},
        'fedproxspoema': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01, 'spo_alpha': 0.05},
        'fedproxspodiv': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 0.01, 'spo_alpha': 0.05},

        # MOON 系列消融
        'moonema': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5},
        'moondiv': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5},
        'moonspoema': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},
        'moonspodiv': {'local_lr': 0.001, 'local_batch_size': 64, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},

        # FedDyn 系列消融
        'feddynema': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01},
        'feddyndiv': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01},
        'feddynspoema': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},
        'feddynspodiv': {'local_lr': 0.001, 'local_batch_size': 64, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},

        # === 其他基线系列 ===
        'fedproto': {'local_lr': 0.001, 'local_batch_size': 64, 'proto_weight': 1.0},
        'fedprotospo': {'local_lr': 0.001, 'local_batch_size': 64, 'proto_weight': 1.0, 'spo_alpha': 0.05},
        'fedprotoheal': {'local_lr': 0.001, 'local_batch_size': 64, 'proto_weight': 1.0},
        'fedprotoshape': {'local_lr': 0.001, 'local_batch_size': 64, 'proto_weight': 1.0, 'spo_alpha': 0.05},

        'ditto': {'local_lr': 0.001, 'local_batch_size': 64, 'ditto_lambda': 1.0},
        'fedfv': {'local_lr': 0.001, 'local_batch_size': 64, 'fv_alpha': 0.1},
        'fdse': {'local_lr': 0.001, 'local_batch_size': 64, 'fdse_lambda': 0.1, 'fdse_beta': 0.001},
        'fedfa': {'local_lr': 0.001, 'local_batch_size': 64, 'fa_weight': 1.0},
        'fedbn': {'local_lr': 0.001, 'local_batch_size': 64},
        'fedlf': {'local_lr': 0.001, 'local_batch_size': 64},
        'feddoga': {'local_lr': 0.001, 'local_batch_size': 64, 'doga_gamma': 2.0, 'doga_clip': 1.0},
        'fedida': {'local_lr': 0.001, 'local_batch_size': 64, 'ida_lambda': 0.1, 'ida_beta': 0.01},
        'fedaa': {'local_lr': 0.001, 'local_batch_size': 64, 'aa_beta': 0.9, 'aa_temp': 1.0},


    },

    'fl_officecaltech': {
        # === 基础 FedAvg 系列 ===
        'fedavg': {'local_lr': 0.001, 'local_batch_size': 16},
        'fedavgspo': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.05},
        'fedavgheal': {'local_lr': 0.001, 'local_batch_size': 16},
        'fedavgshape': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.05},

        # === FedProx 系列 ===
        'fedprox': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01},
        'fedproxspo': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01, 'spo_alpha': 0.05},
        'fedproxheal': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01},
        'fedproxshape': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01, 'spo_alpha': 0.05},

        # === Scaffold 系列 ===
        'scaffold': {'local_lr': 0.001, 'local_batch_size': 16},
        'scaffoldspo': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.05},
        'scaffoldheal': {'local_lr': 0.001, 'local_batch_size': 16},
        'scaffoldshape': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.05},

        # === MOON 系列 ===
        'moon': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5},
        'moonspo': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},
        'moonheal': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5},
        'moonshape': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},

        # === FedDyn 系列 ===
        'feddyn': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01},
        'feddynspo': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},
        'feddynheal': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01},
        'feddynshape': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},

        # === AFL & q-FFL (原版) ===
        'afl': {'local_lr': 0.001, 'local_batch_size': 16, 'afl_lr': 0.01},
        'qffl': {'local_lr': 0.001, 'local_batch_size': 16, 'q_param': 0.1},

        # === AFL/QFFL 组合方法 ===
        'fedavgafl': {'local_lr': 0.001, 'local_batch_size': 16, 'afl_lr': 0.01},
        'fedavgqffl': {'local_lr': 0.001, 'local_batch_size': 16, 'q_param': 0.1},

        'fedproxafl': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01, 'afl_lr': 0.01},
        'fedproxqffl': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01, 'q_param': 0.1},

        'moonafl': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5, 'afl_lr': 0.001},
        'moonqffl': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5, 'q_param': 0.1},

        'feddynafl': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01, 'afl_lr': 0.01},
        'feddynqffl': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01, 'q_param': 0.1},

        # ==========================================
        # 析因消融实验 (Factorial Ablation) 完整列表
        # ==========================================
        'fedsam': {'local_lr': 0.001, 'local_batch_size': 16, 'sam_rho': 0.001},
        'fedsamheal': {'local_lr': 0.001, 'local_batch_size': 16, 'sam_rho': 0.01},
        'spoheal': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.01},

        # FedAvg 系列消融
        'fedavgema': {'local_lr': 0.001, 'local_batch_size': 16},
        'fedavgdiv': {'local_lr': 0.001, 'local_batch_size': 16},
        'fedavgspoema': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.05},
        'fedavgspodiv': {'local_lr': 0.001, 'local_batch_size': 16, 'spo_alpha': 0.05},

        # FedProx 系列消融
        'fedproxema': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01},
        'fedproxdiv': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01},
        'fedproxspoema': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01, 'spo_alpha': 0.05},
        'fedproxspodiv': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 0.01, 'spo_alpha': 0.05},

        # MOON 系列消融
        'moonema': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5},
        'moondiv': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5},
        'moonspoema': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},
        'moonspodiv': {'local_lr': 0.001, 'local_batch_size': 16, 'mu': 5.0, 'temperature': 0.5, 'spo_alpha': 0.05},

        # FedDyn 系列消融
        'feddynema': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01},
        'feddyndiv': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01},
        'feddynspoema': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},
        'feddynspodiv': {'local_lr': 0.001, 'local_batch_size': 16, 'dyn_alpha': 0.01, 'spo_alpha': 0.05},

        # === 其他基线系列 ===
        'fedproto': {'local_lr': 0.001, 'local_batch_size': 16, 'proto_weight': 1.0},
        'fedprotospo': {'local_lr': 0.001, 'local_batch_size': 16, 'proto_weight': 1.0, 'spo_alpha': 0.05},
        'fedprotoheal': {'local_lr': 0.001, 'local_batch_size': 16, 'proto_weight': 1.0},
        'fedprotoshape': {'local_lr': 0.001, 'local_batch_size': 16, 'proto_weight': 1.0, 'spo_alpha': 0.05},

        'ditto': {'local_lr': 0.001, 'local_batch_size': 16, 'ditto_lambda': 1.0},
        'fedfv': {'local_lr': 0.001, 'local_batch_size': 16, 'fv_alpha': 0.1},
        'fdse': {'local_lr': 0.001, 'local_batch_size': 16, 'fdse_lambda': 0.1, 'fdse_beta': 0.001},
        'fedfa': {'local_lr': 0.001, 'local_batch_size': 16, 'fa_weight': 1.0},
        'fedbn': {'local_lr': 0.001, 'local_batch_size': 16},
        'fedlf': {'local_lr': 0.001, 'local_batch_size': 16},
        'feddoga': {'local_lr': 0.001, 'local_batch_size': 16, 'doga_gamma': 2.0, 'doga_clip': 1.0},
        'fedida': {'local_lr': 0.001, 'local_batch_size': 32, 'ida_lambda': 0.1, 'ida_beta': 0.01},
        'fedaa': {'local_lr': 0.001, 'local_batch_size': 32, 'aa_beta': 0.9, 'aa_temp': 1.0},


    }
}
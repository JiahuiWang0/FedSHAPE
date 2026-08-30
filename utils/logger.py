import copy
import os
import csv
from utils.conf import base_path, log_path
from utils.util import create_if_not_exists

useless_args = ['pub_aug', 'public_len', 'public_dataset', 'structure', 'model', 'csv_log', 'device_id', 'seed',
                'tensorboard', 'conf_jobnum', 'conf_timestamp', 'conf_host']
import pickle
import datetime


class CsvWriter:
    def __init__(self, args, private_dataset):
        self.args = args
        self.private_dataset = private_dataset
        self.model_folder_path = self._model_folder_path()
        self.para_foloder_path = self._write_args()
        print(self.para_foloder_path)


        self.run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    def _model_folder_path(self):
        args = self.args
        data_path = log_path() + args.dataset
        create_if_not_exists(data_path)

        model_path = data_path + '/' + args.model
        create_if_not_exists(model_path)
        return model_path

    def generate_filename(self, base_name):
        params_to_include = ['dataset', 'model', 'seed', 'parti_num', 'communication_epoch',
                             'averaging', 'wHEAL', 'threshold', 'beta', 'qfedavg']


        param_strings = []
        for param in params_to_include:
            if hasattr(self.args, param):
                param_strings.append(f"{param}_{getattr(self.args, param)}")


        filename = f"{base_name}_{'_'.join(param_strings)}_{self.run_timestamp}.csv"

        return os.path.join(self.para_foloder_path, filename)

    def write_acc(self, accs_dict, mean_acc_list):
        acc_path = os.path.join(self.para_foloder_path, 'all_acc.csv')
        self._write_all_acc(accs_dict)

        mean_acc_path = os.path.join(self.para_foloder_path, 'mean_acc.csv')
        self._write_mean_acc(mean_acc_list)

    def _write_args(self) -> str:

        args = copy.deepcopy((self.args))
        args = vars(args)
        for cc in useless_args:
            if cc in args:
                del args[cc]

        for key, value in args.items():
            args[key] = str(value)

        # Get existing parameter directories
        paragroup_dirs = []
        if os.path.exists(self.model_folder_path):
            paragroup_dirs = [d for d in os.listdir(self.model_folder_path)
                              if os.path.isdir(os.path.join(self.model_folder_path, d)) and d.startswith('para')]

        n_para = len(paragroup_dirs)
        exist_para = False
        path = None

        # Check if any existing parameter folder has matching args
        for para in paragroup_dirs:
            para_path = os.path.join(self.model_folder_path, para)
            args_path = os.path.join(para_path, 'args.csv')

            # Skip if args.csv doesn't exist
            if not os.path.exists(args_path):
                continue

            try:
                dict_from_csv = {}
                key_value_list = []
                with open(args_path, mode='r', encoding='utf-8') as inp:
                    reader = csv.reader(inp)
                    for rows in reader:
                        key_value_list.append(rows)

                # Check if CSV has both header and data rows
                if len(key_value_list) < 2:
                    continue  # Skip this para if CSV is incomplete

                # Check if header and data rows have the same length
                if len(key_value_list[0]) != len(key_value_list[1]):
                    continue  # Skip if dimensions don't match

                # Build dictionary from CSV
                for index, _ in enumerate(key_value_list[0]):
                    dict_from_csv[key_value_list[0][index]] = key_value_list[1][index]

                # Check if args match
                if args == dict_from_csv:
                    path = para_path
                    exist_para = True
                    break
            except Exception as e:
                # If there's an error reading the file, skip it
                print(f"Warning: Could not read {args_path}: {e}")
                continue

        # Create new parameter folder if no matching args found
        if not exist_para:
            path = os.path.join(self.model_folder_path, 'para' + str(n_para + 1))
            k = 1
            while os.path.exists(path):
                path = os.path.join(self.model_folder_path, 'para' + str(n_para + k))
                k = k + 1
            create_if_not_exists(path)

            # Write args.csv
            columns = list(args.keys())
            args_path = os.path.join(path, 'args.csv')


            os.makedirs(os.path.dirname(args_path), exist_ok=True)

            with open(args_path, 'w', newline='', encoding='utf-8') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                writer.writeheader()
                writer.writerow(args)

        return path

    def _write_mean_acc(self, acc_list):
        mean_path = self.generate_filename('mean_acc')

        os.makedirs(os.path.dirname(mean_path), exist_ok=True)

        with open(mean_path, 'w') as result_file:
            for epoch in range(self.args.communication_epoch):
                result_file.write('epoch_' + str(epoch))
                if epoch != self.args.communication_epoch - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')
            for i in range(len(acc_list)):
                result = acc_list[i]
                result_file.write(str(result))
                if i != self.args.communication_epoch - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

    def _write_all_acc(self, all_acc_list):
        all_path = self.generate_filename('all_acc')


        os.makedirs(os.path.dirname(all_path), exist_ok=True)

        with open(all_path, 'w') as result_file:
            for epoch in range(self.args.communication_epoch):
                result_file.write('epoch_' + str(epoch))
                if epoch != self.args.communication_epoch - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

            for key in all_acc_list:
                method_result = all_acc_list[key]
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

    def write_loss(self, loss_dict, loss_name):

        filename = f"{loss_name}_{self.run_timestamp}.pkl"
        loss_path = os.path.join(self.para_foloder_path, filename)


        os.makedirs(os.path.dirname(loss_path), exist_ok=True)

        with open(loss_path, 'wb+') as f:
            pickle.dump(loss_dict, f)
            f.close()

    def write_round_results(self, epoch_index, domains_list, accs, mean_acc, std):

        filename = f'round_results_{self.run_timestamp}.csv'
        round_results_path = os.path.join(self.para_foloder_path, filename)


        os.makedirs(os.path.dirname(round_results_path), exist_ok=True)


        file_exists = os.path.exists(round_results_path)
        is_first_epoch = (epoch_index == 0)


        mode = 'w' if is_first_epoch else 'a'

        with open(round_results_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)


            if not file_exists or is_first_epoch:
                header = ['epoch'] + [f'{domain}_acc' for domain in domains_list] + ['mean_acc', 'std']
                writer.writerow(header)


            row = [epoch_index] + [acc for acc in accs] + [mean_acc, std]
            writer.writerow(row)

    def write_sys_metrics(self, epoch_index, local_time, peak_mem):


        filename = f'system_metrics_{self.run_timestamp}.csv'
        sys_metrics_path = os.path.join(self.para_foloder_path, filename)


        os.makedirs(os.path.dirname(sys_metrics_path), exist_ok=True)

        file_exists = os.path.exists(sys_metrics_path)
        is_first_epoch = (epoch_index == 0)

        mode = 'w' if is_first_epoch else 'a'

        with open(sys_metrics_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)


            if not file_exists or is_first_epoch:
                header = ['epoch', 'local_time_s', 'peak_vram_mb']
                writer.writerow(header)


            row = [epoch_index, round(local_time, 2), round(peak_mem, 2)]
            writer.writerow(row)
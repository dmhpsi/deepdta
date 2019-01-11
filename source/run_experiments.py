# Thêm thư viện os
import os
# Thêm thư viện tính random
import random as rn
# Thêm thư viện tính thời gian
import time
# Thêm hàm deepcopy từ thư viện copy
from copy import deepcopy

# Thêm thư viện numpy
import numpy as np
# Thêm thư viện pytorch
import torch
# Thêm module nn của pytorch
from torch import nn
# Thêm module optim của pytorch
import torch.optim as optim
# Thêm các module arguments
from arguments import argparser, logging
# Thêm các module của datahelper
from datahelper import *
# Thêm hàm get_cindex của emetrics
from emetrics import get_cindex
# Thêm module functional của module nn của pytorch
from torch.nn import functional

# Cố định hashseed của môi trường
os.environ['PYTHONHASHSEED'] = '0'
# Đặt seed random của numpy
np.random.seed(1)
# Đặt seed random của python
rn.seed(1)

'''
Định nghĩa class DtaNet chứa model
'''


class DtaNet(nn.Module):
    # Phương thức construcotr của class
    def __init__(self, filter_length1, num_filters, filter_length12):
        # Khởi tạo lớp cha
        super(DtaNet, self).__init__()
        # Định nghĩa layer embedding đầu vào
        self.embedding_XD = nn.Embedding(65, 128)
        # Định nghĩa layer convulution thứ nhất
        self.convolution1_XD = nn.Conv1d(128, 32, filter_length1, padding=0)
        # Định nghĩa layer convulution thứ hai
        self.convolution2_XD = nn.Conv1d(num_filters, num_filters * 2, filter_length1, padding=0)
        # Định nghĩa layer convulution thứ ba
        self.convolution3_XD = nn.Conv1d(num_filters * 2, num_filters * 3, filter_length1, padding=0)

        # Định nghĩa lớp embedding đầu vào
        self.embedding_XT = nn.Embedding(26, 128)
        # Định nghĩa layer convulution thứ nhất
        self.convolution1_XT = nn.Conv1d(128, num_filters, filter_length12, padding=0)
        # Định nghĩa layer convulution thứ hai
        self.convolution2_XT = nn.Conv1d(num_filters, num_filters * 2, filter_length12, padding=0)
        # Định nghĩa layer convulution thứ ba
        self.convolution3_XT = nn.Conv1d(num_filters * 2, num_filters * 3, filter_length12, padding=0)

        # Định nghĩa layer fully connected 1
        self.fully_connected1 = nn.Linear(192, 1024)
        # Định nghĩa layer fully connected 2
        self.fully_connected2 = nn.Linear(1024, 1024)
        # Định nghĩa layer fully connected 3
        self.fully_connected3 = nn.Linear(1024, 512)
        # Định nghĩa layer fully connected 4
        self.fully_connected4 = nn.Linear(512, 1)
        # Định nghĩa layer dropout 1
        self.dropout1 = nn.Dropout(0.1)
        # Định nghĩa layer dropout 2
        self.dropout2 = nn.Dropout(0.1)

    # Hàm forward
    def forward(self, XD_input, XT_input):
        # Tính embedding của XD_input
        XD_input = self.embedding_XD(XD_input)
        # Tính layer convulution thứ nhất bằng relu
        smiles = functional.relu(self.convolution1_XD(torch.transpose(XD_input, 2, 1)))
        # Tính layer convulution thứ hai bằng relu
        smiles = functional.relu(self.convolution2_XD(smiles))
        # Tính layer convulution thứ ba bằng relu
        smiles = functional.relu(self.convolution3_XD(smiles))
        # Tính layer max pool 1d
        smiles = functional.max_pool1d(smiles, kernel_size=smiles.size()[2:])
        # Thay đổi shape của kết quả
        smiles = smiles.view(smiles.shape[0], smiles.shape[1])

        # Tính embedding của XT_input
        XT_input = self.embedding_XT(XT_input)
        # Tính layer convulution thứ nhất bằng relu
        protein = functional.relu(self.convolution1_XT(torch.transpose(XT_input, 2, 1)))
        # Tính layer convulution thứ hai bằng relu
        protein = functional.relu(self.convolution2_XT(protein))
        # Tính layer convulution thứ ba bằng relu
        protein = functional.relu(self.convolution3_XT(protein))
        # Tính layer max pool 1d
        protein = functional.max_pool1d(protein, kernel_size=protein.size()[2:])
        # Thay đổi shape của kết quả
        protein = protein.view(protein.shape[0], protein.shape[1])

        # Gộp hai layer
        interaction = torch.cat((smiles, protein), 1)
        # Tính layer fully connected 1 bằng relu
        f_relu = functional.relu(self.fully_connected1(interaction))
        # Tính dropout 1
        f_relu = self.dropout1(f_relu)
        # Tính layer fully connected 2 bằng relu
        f_relu = functional.relu(self.fully_connected2(f_relu))
        # Tính dropout 12
        f_relu = self.dropout2(f_relu)
        # Tính layer fully connected 3 bằng relu
        f_relu = functional.relu(self.fully_connected3(f_relu))
        # Tính layer fully connected 4
        f_relu = self.fully_connected4(f_relu)
        # Trả về kết quả
        return f_relu


# Hàm xử lý dữ liệu training
def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, FLAGS, dataset):
    # Đọc dữ liệu
    test_set, outer_train_sets = dataset.read_sets(FLAGS.dataset_path, FLAGS.problem_type)
    # Tính số lượng fold
    foldinds = len(outer_train_sets)

    # Khởi tạo test_sets
    test_sets = []
    # Khởi tạo val_sets
    val_sets = []
    # Khởi tạo train_sets
    train_sets = []

    # Duyệt từng fold
    for val_foldind in range(foldinds):
        # Lấy val_fold từ outer_train_sets
        val_fold = outer_train_sets[val_foldind]
        # Thêm vào val_sets
        val_sets.append(val_fold)
        # Sao chép outer_train_sets
        otherfolds = deepcopy(outer_train_sets)
        # Xóa entry cuối của otherfolds
        otherfolds.pop(val_foldind)
        # Tạo list các item trong otherfolds
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        # Thêm list vừa tạo vào train_sets
        train_sets.append(otherfoldsinds)
        # Thêm test_set vào test_sets
        test_sets.append(test_set)
        # In val_fold ra màn hình
        print("val set", str(len(val_fold)))
        # In otherfoldsinds ra màn hình
        print("train set", str(len(otherfoldsinds)))

    # Tính hyperparameter của train_sets và val_sets
    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(
        XD, XT, Y,
        label_row_inds,
        label_col_inds,
        measure,
        FLAGS,
        train_sets,
        val_sets)

    # Tính hyperparameter của train_sets và test_sets
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(
        XD, XT, Y,
        label_row_inds,
        label_col_inds,
        measure,
        FLAGS,
        train_sets,
        test_sets)

    # Log file
    logging("---FINAL RESULTS-----", FLAGS)
    # Log best param vào file log
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)
    # In ra màn hình
    print("---FINAL RESULTS-----")
    # In best param ra màn hình
    print("best param index = %s,  best param = %.5f" %
          (bestparamind, bestparam))

    # Khởi tạo testperfs
    testperfs = []
    # Khởi tạo testloss
    testloss = []

    # Khỏi tạo avgperf
    avgperf = 0.

    # Duyệt từng test_set
    for test_foldind in range(len(test_sets)):
        # Lấy test performance CI
        foldperf = all_predictions[bestparamind][test_foldind]
        # Lấy test performance MSE
        foldloss = all_losses[bestparamind][test_foldind]
        # Thêm vào testperfs
        testperfs.append(foldperf)
        # Thêm vào testloss
        testloss.append(foldloss)
        # Tính avgperf
        avgperf += foldperf

    # Tính avgperf
    avgperf = avgperf / len(test_sets)
    # Tính trung bình loss
    avgloss = np.mean(testloss)
    # Tính độ lệch chuẩn loss
    teststd = np.std(testperfs)

    # Log vào file
    logging("Test Performance CI", FLAGS)
    # Log vào file
    logging(testperfs, FLAGS)
    # Log vào file
    logging("Test Performance MSE", FLAGS)
    # Log vào file
    logging(testloss, FLAGS)

    # Trả về giá trị
    return avgperf, avgloss, teststd


# Hàm tính cross fold validation
def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, FLAGS, labeled_sets,
                     val_sets):
    # Tách parameter1 từ FLAGS
    paramset1 = FLAGS.num_windows  # [32]#[32,  512] #[32, 128]  # filter numbers
    # Tách parameter2 từ FLAGS
    paramset2 = FLAGS.smi_window_lengths  # [4, 8]#[4,  32] #[4,  8] #filter length smi
    # Tách parameter3 từ FLAGS
    paramset3 = FLAGS.seq_window_lengths  # [8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    # Tách parameter4 từ FLAGS
    epoch = FLAGS.num_epoch  # 100
    # Tách batch size từ FLAGS
    batchsz = FLAGS.batch_size  # 256

    # Log vào file
    logging("---Parameter Search-----", FLAGS)
    # Log vào file
    print("---Parameter Search-----")

    # Tính kích thước val_sets
    w = len(val_sets)
    # Tính kích thước
    h = len(paramset1) * len(paramset2) * len(paramset3)

    # Khởi tạo list all_predictions
    all_predictions = [[0 for x in range(w)] for y in range(h)]
    # Khởi tạo list all_losses
    all_losses = [[0 for x in range(w)] for y in range(h)]

    # Duyệt từng chỉ số foldind
    for foldind in range(len(val_sets)):
        # Lấy giá trị valinds từ foldind
        valinds = val_sets[foldind]
        # Lấy giá trị valinds từ foldind
        labeledinds = labeled_sets[foldind]

        # Lấy giá trị trrows từ label_row_inds
        trrows = label_row_inds[labeledinds]
        # Lấy giá trị trcols từ label_col_inds
        trcols = label_col_inds[labeledinds]

        # Lấy các tập train
        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        # Lấy giá trị terows từ label_row_inds
        terows = label_row_inds[valinds]
        # Lấy giá trị tecols từ label_col_inds
        tecols = label_col_inds[valinds]

        # Lấy các tập validation
        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        # Khởi tạo biến đếm vòng lặp
        pointer = 0

        # Duyệt từng chỉ số param1ind
        for param1ind in range(len(paramset1)):
            # Lấy giá trị từ paramset1
            param1value = paramset1[param1ind]

            # Duyệt từng chỉ số param2ind
            for param2ind in range(len(paramset2)):
                # Lấy giá trị từ paramset2
                param2value = paramset2[param2ind]

                # Duyệt từng chỉ số param3ind
                for param3ind in range(len(paramset3)):
                    # Lấy giá trị từ paramset3
                    param3value = paramset3[param3ind]

                    # Khởi tạo model
                    dta_model = DtaNet(param2value, param1value, param3value)
                    # Chạy trên GPU
                    dta_model.cuda()
                    # Khởi tạo hàm loss
                    loss_func = nn.MSELoss()
                    # Khởi tạo optimizer
                    optimizer = optim.Adam(dta_model.parameters(), lr=0.0005)
                    # Khởi tạo predicted_labels
                    predicted_labels = []

                    # Duyệt từng epoch
                    for i in range(epoch):
                        # Khởi tạo loss_epoch
                        loss_epoch = 0
                        # Bắt đầy training
                        dta_model.train()
                        # Tính số lượng train_drugs
                        train_drugs_count = len(train_drugs)
                        # Duyệt các bộ dữ liệu
                        for j in range(0, train_drugs_count, batchsz):
                            # Xóa cache cuda
                            torch.cuda.empty_cache()
                            # Reset gradient
                            optimizer.zero_grad()
                            # Tính dữ liệu
                            k = min(j + batchsz, train_drugs_count)
                            # Lấy bộ dữ liệu con
                            sub_train_drugs = train_drugs[j:k]
                            # Lấy bộ dữ liệu con
                            sub_train_prots = train_prots[j:k]
                            # Lấy bộ dữ liệu con
                            target = train_Y[j:k]
                            # Chuyển thành tensor
                            target = torch.FloatTensor(target)
                            # Sử dụng cuda
                            target = target.cuda()
                            # Chuyển thành tensor
                            sub_train_drugs = torch.tensor(sub_train_drugs, dtype=torch.long)
                            # Sử dụng cuda
                            sub_train_drugs = sub_train_drugs.cuda()
                            # Chuyển thành tensor
                            sub_train_prots = torch.tensor(sub_train_prots, dtype=torch.long)
                            # Sử dụng cuda
                            sub_train_prots = sub_train_prots.cuda()
                            # Tính dự đoán
                            output = dta_model(sub_train_drugs, sub_train_prots)
                            # Khởi tạo hàm loss
                            loss = loss_func(output, target)
                            # Tính gradient theo parameter
                            loss.backward()
                            # Back propagation
                            optimizer.step()
                            # Tính tổng loss
                            loss_epoch += loss.item() * len(sub_train_drugs)

                        # Chuyển sang chế độ đánh giá
                        dta_model.eval()
                        # Khởi tạo loss_eval
                        loss_eval = 0
                        # Duyệt các bộ dữ liệu
                        for j in range(0, int(len(val_drugs)), batchsz):
                            # Xóa cache cuda
                            torch.cuda.empty_cache()
                            # Tính dữ liệu
                            k = min(j + batchsz, len(val_drugs))
                            # Lấy bộ dữ liệu con
                            sub_train_drugs = val_drugs[j:k]
                            # Lấy bộ dữ liệu con
                            sub_train_prots = val_prots[j:k]
                            # Lấy bộ dữ liệu con
                            target = val_Y[j:k]
                            # Chuyển thành tensor
                            target = torch.FloatTensor(target)
                            # Sử dụng cuda
                            target = target.cuda()
                            # Chuyển thành tensor
                            sub_train_drugs = torch.tensor(sub_train_drugs, dtype=torch.long)
                            # Sử dụng cuda
                            sub_train_drugs = sub_train_drugs.cuda()
                            # Chuyển thành tensor
                            sub_train_prots = torch.tensor(sub_train_prots, dtype=torch.long)
                            # Sử dụng cuda
                            sub_train_prots = sub_train_prots.cuda()
                            # Tính dự đoán
                            output = dta_model(sub_train_drugs, sub_train_prots)
                            # Tính loss
                            loss = loss_func(output, target)
                            # Tính tổng loss
                            loss_eval += loss.item() * len(sub_train_drugs)
                            # Nếu là epoch cuối
                            if i == epoch - 1:
                                # Nếu kết quả phù hợp
                                if len(predicted_labels) == 0:
                                    # Chuyển về numpy
                                    predicted_labels = output.cpu().detach().numpy()
                                else:
                                    # Chuyển về numpy
                                    predicted_labels = np.concatenate((predicted_labels, output.cpu().detach().numpy()),
                                                                      0)
                        # In ra màn hình
                        print("epoch #", i + 1, ", train loss", loss_epoch * 1.0 / len(train_drugs),
                              ", validation loss", loss_eval / len(val_drugs))
                        # Log vào file
                        logging("epoch #" + str(i + 1) + ", train loss " + str(
                            loss_epoch * 1.0 / len(train_drugs)) + ", validation loss " + str(
                            loss_eval / len(val_drugs)), FLAGS)

                    # Tính CI-i
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]

                    # Log vào file
                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, MSE = %f" %
                            (param1ind, param2ind, param3ind, foldind, rperf, loss_eval / len(val_drugs)), FLAGS)
                    # In ra màn hình
                    print("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, MSE = %f" %
                          (param1ind, param2ind, param3ind, foldind, rperf, loss_eval / len(val_drugs)))

                    # Lưu lại chỉ số rperf
                    all_predictions[pointer][
                        foldind] = rperf
                    # Lưu lại loss
                    all_losses[pointer][foldind] = loss_eval / len(val_drugs)
                    # Tăng pointer thêm 1
                    pointer += 1
    # Khởi tạo bestperf
    bestperf = -float('Inf')
    # Khởi tạo bestpointer
    bestpointer = None

    # Khởi tạo best_param_list
    best_param_list = []
    # Khởi tạo biến đếm
    pointer = 0

    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                # Tính avgPerf trung bình
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                # avgPerf tốt hơn đã ghi nhận
                if avgperf > bestperf:
                    # Lưu lại
                    bestperf = avgperf
                    # Lưu lại
                    bestpointer = pointer
                    # Lưu lại
                    best_param_list = [param1ind, param2ind, param3ind]
                # Tăng biến đếm thêm 1
                pointer += 1
    # Trả về các giá trị
    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    # Khởi tạo drugs
    drugs = []
    # Khởi tạo biến targets
    targets = []
    # Khởi tọa biến affinity
    affinity = []

    # Duyệt từng pair_ind
    for pair_ind in range(len(rows)):
        # Lấy giá trị drug
        drug = XD[rows[pair_ind]]
        # Thêm vào list drugs
        drugs.append(drug)

        # Lấy giá trị target
        target = XT[cols[pair_ind]]
        # Thêm vào targets
        targets.append(target)
        # Thêm vào affinity
        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, perfmeasure, foldcount=6):  # 5-fold cross validation + test

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,  # BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.problem_type,  # BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(fpath=FLAGS.dataset_path)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,Y]

    print("Logdir: " + FLAGS.log_dir)
    s1_avgperf, s1_avgloss, s1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    perfmeasure, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (s1_avgperf, s1_avgloss, s1_teststd), FLAGS)
    print("Setting " + str(FLAGS.problem_type))
    print("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
          (s1_avgperf, s1_avgloss, s1_teststd))


def run_regression(FLAGS):
    perfmeasure = get_cindex
    experiment(FLAGS, perfmeasure)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    print(str(FLAGS))
    run_regression(FLAGS)

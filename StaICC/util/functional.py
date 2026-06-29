import math
import warnings
from . import configs

def probability_matrix_check(x):
    if not all([probability_vector_check(x_i) for x_i in x]):
        return False
    return True

def probability_vector_check(x):
    if not all([0 <= x_i <= 1 for x_i in x]):
        return False
    if sum(x) != 1:
        return False
    return True

def exp_to_list(list):
    return [math.exp(x) for x in list]

def L2_dist(x, y):
    if len(x) != len(y):
        raise ValueError("The length of x and y should be the same.")
    return sum([(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)]) ** 0.5

def linear_regression(x, y):
    if len(x) != len(y):
        raise ValueError("The length of x and y should be the same.")
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_square = sum([x_i ** 2 for x_i in x])
    sum_xy = sum([x_i * y_i for x_i, y_i in zip(x, y)])
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_square - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b

def two_dimensional_list_mean(x):
    sum = [0] * len(x[0])
    for x_i in x:
        for i in range(len(x_i)):
            sum[i] += x_i[i]
    return [x_i / len(x) for x_i in sum]

def bias_mean_metric(ground_truth, prediction):
    return two_dimensional_list_mean(prediction)

def bias_mean_entropy_metric(ground_truth, prediction):
    return entropy(two_dimensional_list_mean(prediction))

def post_bias_dis_metric(ground_truth, prediction):
    label_dis = [0] * len(prediction[0])
    for i in range(len(ground_truth)):
        label_dis[ground_truth[i]] += 1
    label_dis = [x / len(ground_truth) for x in label_dis]
    averaged_prediction = two_dimensional_list_mean(prediction)
    return [prediction_i - label_dis_i for prediction_i, label_dis_i in zip(averaged_prediction, label_dis)]

def kl_divergence(x, y):
    if len(x) != len(y):
        raise ValueError("The length of x and y should be the same.")
    return sum([x_i * math.log(x_i / y_i) for x_i, y_i in zip(x, y)])

def post_bias_dl_metric(ground_truth, prediction):
    label_dis = [0] * len(prediction[0])
    for i in range(len(ground_truth)):
        label_dis[ground_truth[i]] += 1
    label_dis = [x / len(ground_truth) for x in label_dis]
    averaged_prediction = two_dimensional_list_mean(prediction)
    return kl_divergence(averaged_prediction, label_dis)

def softmax(x):
    f_x_max = max(x)
    f_x = [x_i - f_x_max for x_i in x]
    f_x = exp_to_list(f_x) 
    sum_x = sum(f_x)
    return [x_i / sum_x for x_i in f_x]

def entropy(x): # nats
    # if not all([0 <= x_i <= 1 for x_i in x]):
    if not probability_vector_check(x):
        x = softmax(x)
    if type(x[0]) != list:
        return -sum([x_i * math.log(x_i) for x_i in x if x_i != 0])
    else:
        return entropy(two_dimensional_list_mean(x))

def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])

def argmin(x):
    return min(range(len(x)), key=lambda i: x[i])

def unique_check(list):
    if len(list) != len(set(list)):
        return False
    else:
        return True

def linspace(start, end, num):
    if num <= 1:
        raise ValueError("num should be greater than 1.")
    return [start + (end - start) * i / (num - 1) for i in range(num)]

def extend_onehot_prediction_to_logits(prediction: list[int]) -> list[list[float]]:
    if type(prediction[0]) == list:
        return prediction
    return [[1 if i == x else 0 for i in range(max(prediction) + 1)] for x in prediction]
    
def compress_logits_prediction_to_onehot(prediction: list[list[float]]) -> list[int]:
    if type(prediction[0]) == int:
        return prediction
    return [argmax(x) for x in prediction]

def accuracy(ground_truth: list[int], prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    # if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
    if not probability_matrix_check(prediction):
        prediction = [softmax(x) for x in prediction]
    
    correct = 0
    for i in range(len(ground_truth)):
        if argmax(prediction[i]) == ground_truth[i]:
            correct += 1
    return correct / len(ground_truth)


def averaged_truelabel_likelihood(ground_truth: list[int], prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    # if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
    if not probability_matrix_check(prediction):
        prediction = [softmax(x) for x in prediction]
    
    likelihood = 0
    for i in range(len(ground_truth)):
        likelihood += prediction[i][ground_truth[i]]
    return likelihood / len(ground_truth)


def macro_F1(ground_truth: list[int], prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    # if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
    if not probability_matrix_check(prediction):
        prediction = [softmax(x) for x in prediction]
    
    TP = [0] * len(prediction[0])
    FP = [0] * len(prediction[0])
    FN = [0] * len(prediction[0])
    for i in range(len(ground_truth)):
        if argmax(prediction[i]) == ground_truth[i]:
            TP[ground_truth[i]] += 1
        else:
            FP[argmax(prediction[i])] += 1
            FN[ground_truth[i]] += 1
    
    precision = [TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0 for i in range(len(TP))]
    recall = [TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) != 0 else 0 for i in range(len(TP))]
    F1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 for i in range(len(TP))]
    return sum(F1) / len(F1)


def expected_calibration_error_1(ground_truth: list[int], prediction, bins = configs.STANDARD_SETTINGS["ece_bins"]):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    if not probability_matrix_check(prediction):
        prediction = [softmax(x) for x in prediction]
    if bins <= 1:
        raise ValueError("bins should be greater than 1.")
    if bins != configs.STANDARD_SETTINGS["ece_bins"]:
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["strict_mode_protect"])
            bins = configs.STANDARD_SETTINGS["ece_bins"]
    if bins > len(ground_truth):
        raise ValueError("bins should be less than the length of ground_truth.")
    
    bin_boundaries = linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = [max(x) for x in prediction]
    predicted_label = [argmax(x) for x in prediction]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [confidence for confidence, label in zip(confidences, predicted_label) if bin_lower <= confidence < bin_upper]
        if len(in_bin) == 0:
            continue
        accuracy_in_bin = [1 if label == ground_truth[i] else 0 for i, label in enumerate(predicted_label) if bin_lower <= confidences[i] < bin_upper]
        ece += len(in_bin) / len(ground_truth) * abs(sum(accuracy_in_bin) / len(accuracy_in_bin) - sum(in_bin) / len(in_bin))
    return ece


def single_consistency(prediction: list[int]):
    majority = max(set(prediction), key=prediction.count)
    return prediction.count(majority) / len(prediction)


def consistency(ground_truth, prediction, loop_length = 100):
    if not probability_matrix_check(prediction):
        prediction = [softmax(x) for x in prediction]
    if len(prediction) % loop_length != 0:
        raise ValueError("The length of prediction should be the multiple of loop_length.")
    
    sum = 0
    for i in range(loop_length):
        single_prediction = []
        for j in range(i, len(prediction), loop_length):
            single_prediction.append(argmax(prediction[j]))
        sum += single_consistency(single_prediction)
    return sum / loop_length
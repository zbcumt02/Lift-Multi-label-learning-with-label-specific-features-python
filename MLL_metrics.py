import torch


def hamming_loss(test_target, pre_Labels):
    num_class, num_instance = pre_Labels.shape
    miss_pairs = (pre_Labels != test_target).sum().item()
    hamming_loss = miss_pairs / (num_class * num_instance) * 1.0
    return hamming_loss


def label_index_preprocessing(outputs, test_target):
    num_class, num_instance = outputs.shape

    label = []
    not_label = []
    label_size = torch.zeros(num_instance)

    for i in range(num_instance):
        temp = test_target[:, i]
        label_size[i] = (temp == 1).sum().item()  # pos label num
        label_indices = (temp == 1).nonzero().squeeze().tolist()
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        not_label_indices = (temp != 1).nonzero().squeeze().tolist()
        if isinstance(not_label_indices, int):
            not_label_indices = [not_label_indices]

        label.append(label_indices)
        not_label.append(not_label_indices)

    return num_class, num_instance, label, not_label, label_size


def mask_preprocessing(outputs, test_target):
    num_class, num_instance = outputs.shape

    # mask all 1 or all !1
    mask = (test_target.sum(dim=0) != num_class) & (test_target.sum(dim=0) != -num_class)
    outputs = outputs[:, mask]
    test_target = test_target[:, mask]

    return outputs, test_target


def average_precision(outputs, test_target):
    outputs, test_target = mask_preprocessing(outputs, test_target)
    num_class, num_instance, label, not_label, label_size = label_index_preprocessing(outputs, test_target)

    aveprec = 0
    for i in range(num_instance):
        temp = outputs[:, i]

        # sort for output
        sorted_values, sorted_indices = torch.sort(temp, descending=False)

        indicator = torch.zeros(num_class)

        # pos label index
        for label_idx in label[i]:
            indicator[sorted_indices == label_idx] = 1

        summary = 0
        # calculate average precision for all instance
        for m in range(label_size[i].int()):
            label_idx = label[i][m]
            loc = (sorted_indices == label_idx).nonzero().item()  # label location
            summary += indicator[loc:].sum().item() / (num_class - loc)

        ap_binary = summary / label_size[i].item() if label_size[i] > 0 else 0
        aveprec += ap_binary

    average_precision = aveprec / num_instance
    return average_precision


def coverage(outputs, test_target):
    num_class, num_instance, label, not_label, label_size = label_index_preprocessing(outputs, test_target)
    # cal coverage
    cover = 0
    for i in range(num_instance):
        temp = outputs[:, i]
        # sort output & index
        _, index = torch.sort(temp, descending=False)
        temp_min = num_class
        for m in range(int(label_size[i])):
            label_index = label[i][m]
            loc = torch.where(index == label_index)[0].item()
            if loc < temp_min:
                temp_min = loc
        cover += (num_class - temp_min)
    coverage_value = ((cover / num_instance) - 1) / num_class
    return coverage_value


def one_error(outputs, test_target):
    outputs, test_target = mask_preprocessing(outputs, test_target)
    num_class, num_instance, label, not_label, label_size = label_index_preprocessing(outputs, test_target)

    one_error = 0
    for i in range(num_instance):
        indicator = 0
        temp = outputs[:, i]
        maximum, index = temp.max(dim=0)  # max for comparison
        for j in range(num_class):
            if temp[j] == maximum.item():
                if j in label[i]:
                    indicator = 1
                    break
        if indicator == 0:
            one_error += 1

    one_error = one_error / num_instance
    return one_error


def ranking_loss(outputs, test_target):
    outputs, test_target = mask_preprocessing(outputs, test_target)
    num_class, num_instance, label, not_label, label_size = label_index_preprocessing(outputs, test_target)

    ranking_loss = 0.0
    for i in range(num_instance):
        temp = 0
        for m in range(int(label_size[i])):
            for n in range(num_class - int(label_size[i])):
                if outputs[label[i][m], i] <= outputs[not_label[i][n], i]:
                    temp += 1
        rl_binary = temp / (label_size[i] * (num_class - label_size[i]))
        ranking_loss += rl_binary

    ranking_loss = ranking_loss / num_instance
    return ranking_loss



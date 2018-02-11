import os
import json
import numpy as np

def class_recall(pred_classes, true_classes, class_label):

    if class_label == "positive":
        class_id = 0
    elif class_label == "neutral":
        class_id = 1
    elif class_label == "negative":
        class_id = 2
    else:
        print("error in classes")

    tp = 0 # true positive
    for i in range(len(pred_classes)):
        if pred_classes[i] == true_classes[i]:
            if pred_classes[i] == class_id:
                tp += 1

    return tp/(true_classes.count(class_id))


def avg_recall_on_training_end(auxOutputsFlag, model, x_test, y_test, features_test, pos_test, stanford_test, lexicon_test):
        output = model.predict([x_test, features_test, pos_test, stanford_test, lexicon_test])
        pred_classes = []
        true_classes = []



        for i in range(x_test.shape[0]):
            if auxOutputsFlag:
                pred_classes.append(output[0][i].argmax())
            else:
                pred_classes.append(output[i].argmax())
            true_classes.append(y_test[i].argmax())

        #douleuei kai me auto!! alla ithela na vlepo ta epimerouw recalls ton classeon
        # print("sklearn.metrics: recall_score = " + str(recall_score(true_classes, pred_classes, average='macro')))

        recall_p = class_recall(pred_classes, true_classes, class_label="positive")
        recall_u = class_recall(pred_classes, true_classes, class_label="neutral")
        recall_n = class_recall(pred_classes, true_classes, class_label="negative")

        print("Positive recall: " + str(recall_p))
        print("Neutral recall: " + str(recall_u))
        print("Negative recall: " + str(recall_n))

        avg_rec = (recall_p + recall_u + recall_n) / 3

        return avg_rec, recall_p, recall_u, recall_n


def log_to_json(dim, epochs, max_seq_len, max_nb_words, batch, seed, tool, auxOutputsFlag, attention,
                nb_filters, kernel, pos_kernel, max_pool, recall, recalls, noise,
                drop_text_input, drop_castle, drop_emb_tower, file, optimizer, l1, l2, less_data):

    if not os.path.isfile(file):
        open(file, 'a').close()

    with open(file, mode='r', encoding='utf-8') as feedsjson:
        try:
            feeds = json.load(feedsjson)

        except:
            feeds = []

    with open(file, mode='w', encoding='utf-8') as feedsjson:
        entry = {'EmbeddingsDim': dim,
                 'Epochs(Stopped)': epochs,
                 'MaxSequenceLength': max_seq_len,
                 'MaxNbWords': max_nb_words,
                 'BatchSize': batch,
                 'Seed': seed,
                 'Embedding': tool,
                 'AuxiliaryOutputs': auxOutputsFlag,
                 'Attention': attention,
                 'NbFilters': nb_filters,
                 'Optimizer': optimizer,
                 'FilterKernelSize': kernel,
                 'POSFilterKernelSize': pos_kernel,
                 'MaxPoolingWindow': max_pool,
                 'NoiseInput': noise,
                 'L1': l1,
                 'L2': l2,
                 'LessData': less_data,
                 'DropTextInput': drop_text_input,
                 'DropCastle': drop_castle,
                 'DropEmbTower': drop_emb_tower,
                 '_PositiveRecall': recalls[0],
                 '_NeutralRecall': recalls[1],
                 '_NegativeRecall': recalls[2],
                 '__AVG_RECALL': recall}
        feeds.append(entry)
        json.dump(feeds, feedsjson, sort_keys=True, indent=2)

    print("Logged to file")


def save_weights(model, seed, epoch, recall, floydhub):

    if floydhub:
        filename = "/output/model" + str(seed) + "epoch" + str(epoch + 1) + "avg_recall" + str(recall) + ".h5"

    else:
        filename = "output/model" + str(seed) + "epoch" + str(epoch + 1) + "avg_recall" + str(recall) + ".h5"


    model.save_weights(filename)


def load(floydhub, less_data):
    # load data and labels from file
    print("loading data and labels")
    if floydhub:
        data_train_ = np.load("/my_data/data_TRAIN.npy")
        data_test_ = np.load("/my_data/data_TEST.npy")
        labels_train_ = np.load("/my_data/labels_TRAIN.npy")
        labels_test_ = np.load("/my_data/labels_TEST.npy")
        features_train_ = np.load("/my_data/features_TRAIN_norm_ourliered.npy")
        features_test_ = np.load("/my_data/features_TEST_norm_ourliered.npy")
        embedding_matrix_glove_ = np.load("/my_data/embeddingMatrix_gloveTRAIN_n_TEST_DATA300ALL-50-50000.npy")
        embedding_matrix_word2vec_ = np.load("/my_data/embeddingMatrix_word2vecTRAIN_n_TEST_DATA300ALL-50-50000.npy")
        pos_train_ = np.load("/my_data/pos_TRAIN.npy")
        pos_test_ = np.load("/my_data/pos_TEST.npy")
        pos_embedding_matrix_ = np.load("/my_data/pos_emb_matrix.npy")
        stanford_train_ = np.load("/my_data/stanford_TRAIN.npy")
        stanford_test_ = np.load("/my_data/stanford_TEST.npy")
        lexicons_train_ = np.load("/my_data/lexicons_TRAIN_norm.npy")
        lexicons_test_ = np.load("/my_data/lexicons_TEST_norm.npy")
    else:
        data_train_ = np.load("data/data_TRAIN.npy")
        data_test_ = np.load("data/data_TEST.npy")
        labels_train_ = np.load("data/labels_TRAIN.npy")
        labels_test_ = np.load("data/labels_TEST.npy")
        features_train_ = np.load("data/features_TRAIN_norm_ourliered.npy")
        features_test_ = np.load("data/features_TEST_norm_ourliered.npy")
        embedding_matrix_glove_ = np.load("data/embeddingMatrix_gloveTRAIN_n_TEST_DATA300ALL-50-50000.npy")
        embedding_matrix_word2vec_ = np.load("data/embeddingMatrix_word2vecTRAIN_n_TEST_DATA300ALL-50-50000.npy")
        pos_train_ = np.load("data/pos_TRAIN.npy")
        pos_test_ = np.load("data/pos_TEST.npy")
        pos_embedding_matrix_ = np.load("data/pos_emb_matrix.npy")
        stanford_train_ = np.load("data/stanford_TRAIN.npy")
        stanford_test_ = np.load("data/stanford_TEST.npy")
        lexicons_train_ = np.load("data/lexicons_TRAIN_norm.npy")
        lexicons_test_ = np.load("data/lexicons_TEST_norm.npy")


    print("Maximum of features array: %s" % features_train_.max())

    if less_data:
        return data_train_[0:1000], \
               data_test_[0:100], \
               labels_train_[0:1000], \
               labels_test_[0:100], \
               features_train_[0:1000], \
               features_test_[0:100], \
               embedding_matrix_glove_, \
               embedding_matrix_word2vec_, \
               pos_train_[0:1000], \
               pos_test_[0:100], \
               pos_embedding_matrix_, \
               stanford_train_[0:1000], \
               stanford_test_[0:100],  \
               lexicons_train_[0:1000], \
               lexicons_test_[0:100]

    else:
        return data_train_, data_test_, labels_train_, labels_test_, features_train_, features_test_, embedding_matrix_glove_, embedding_matrix_word2vec_, pos_train_, pos_test_, pos_embedding_matrix_, stanford_train_, stanford_test_, lexicons_train_, lexicons_test_


def split_data(data_, labels_, features_):
    # split the data into a training set and a validation set
    indices = np.arange(data_.shape[0])
    np.random.shuffle(indices)
    data_ = data_[indices]
    labels_ = labels_[indices]
    features_ = features_[indices]

    num_validation_samples = int(VALIDATION_SPLIT * data_.shape[0])

    x_train_ = data_[:-num_validation_samples]
    y_train_ = labels_[:-num_validation_samples]
    x_val_ = data_[-num_validation_samples:]
    y_val_ = labels_[-num_validation_samples:]
    features_train_ = features_[:-num_validation_samples]
    features_val_ = features_[-num_validation_samples:]

    return x_train_, y_train_, x_val_, y_val_, features_train_, features_val_

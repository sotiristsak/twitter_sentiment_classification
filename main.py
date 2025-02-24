import numpy as np
from modelUtils import *
from utils import *
from keras import initializers


if __name__ == "__main__":

    # run on floyd
    # floyd run --gpu --data aleeex32/datasets/glove300allfeatures/21:/my_data python main.py

    floydhub = False  # for running on Floydhub.com
    less_data = False
    auxOutputs = True
    attention = True
    saveWeights = False

    if floydhub:
        JSON_FILE = "/output/runs.json"
    else:
        JSON_FILE = "output/runs.json"



    # do not change
    MAX_SEQUENCE_LENGTH = 50
    MAX_NB_WORDS = 50000
    OUTPUT_DIM = 3
    EMBEDDING_DIM = 300



    #constants
    EMBEDDING_TOOL = "Word2Vec"
    POS_FILTER_KERNEL_SIZE = [2, 3, 4]
    NOISE_INPUT = 0.02
    DROP_TEXT_INPUT = 0.4
    DROP_CASTLE = 0.2
    DROP_EMB_TOWER = 0.2
    OPTIMIZER = "adadelta"  # sgd, rmsprop, adagrad, adadelta, adamadamax, nadam, tfoptimizer, adam
    # L1 = 0.01  # regularization
    L2 = 0.01
    # INITIALIZER = initializers.RandomNormal(0, stddev=0.01)

    #you can adjust them!
    list_SEED = [11]
    list_NUMBER_OF_FILTERS = [256]
    list_FILTER_KERNEL_SIZE = [5]
    list_MAX_POOLING_WINDOW = [5]
    list_MIN_IMPROVEMENT = [0.00001]
    list_IMPROVEMENT_PATIENCE = [30]
    list_EPOCHS = [200]
    list_BATCH_SIZE = [5000]
    list_L1 = [4]


    [data_train, data_test, labels_train, labels_test, features_train, features_test, embedding_matrix_glove, embedding_matrix_word2vec, pos_train, pos_test, pos_embedding_matrix, stanford_train, stanford_test, lexicons_train, lexicons_test] = load(floydhub, less_data)

    # [x_train, y_train, x_val, y_val, features_train, features_val] = split_data(data, labels, features)


    for SEED, NUMBER_OF_FILTERS, FILTER_KERNEL_SIZE, MAX_POOLING_WINDOW, MIN_IMPROVEMENT, IMPROVEMENT_PATIENCE, EPOCHS, BATCH_SIZE, L1 in \
            [(SEED, NUMBER_OF_FILTERS, FILTER_KERNEL_SIZE, MAX_POOLING_WINDOW, MIN_IMPROVEMENT,
              IMPROVEMENT_PATIENCE, EPOCHS, BATCH_SIZE, L1)
             for SEED in list_SEED
             for NUMBER_OF_FILTERS in list_NUMBER_OF_FILTERS
             for FILTER_KERNEL_SIZE in list_FILTER_KERNEL_SIZE
             for MAX_POOLING_WINDOW in list_MAX_POOLING_WINDOW
             for MIN_IMPROVEMENT in list_MIN_IMPROVEMENT
             for IMPROVEMENT_PATIENCE in list_IMPROVEMENT_PATIENCE
             for EPOCHS in list_EPOCHS
             for BATCH_SIZE in list_BATCH_SIZE
             for L1 in list_L1]:

        np.random.seed(SEED)  # for reproducibility

        model = create_model(emb_mat_glove=embedding_matrix_glove,
                             emb_mat_w2v=embedding_matrix_word2vec,
                             emb_dim=EMBEDDING_DIM,
                             emb_mat_pos=pos_embedding_matrix,
                             emb_pos_dim=pos_embedding_matrix.shape[1],
                             max_seq_len=MAX_SEQUENCE_LENGTH,
                             max_pool_win=MAX_POOLING_WINDOW,
                             nb_filters=NUMBER_OF_FILTERS,
                             filter_kernel=FILTER_KERNEL_SIZE,
                             pos_filter_kernel=POS_FILTER_KERNEL_SIZE,
                             out_dim=OUTPUT_DIM,
                             features_len=features_train.shape[1],
                             lexicons_len=lexicons_train.shape[1],
                             pos_len=pos_train.shape[1],
                             noise=NOISE_INPUT,
                             drop_text_input=DROP_TEXT_INPUT,
                             drop_emb_tower=DROP_EMB_TOWER,
                             drop_castle=DROP_CASTLE,
                             l1=L1,
                             l2=L2,
                             stanford_shape=stanford_train.shape,
                             attentionFlag=attention,
                             auxOutputsFlag=auxOutputs)

        list_of_avg_recalls, list_of_recalls = train_model(model,
                                                           data_train,
                                                           labels_train,
                                                           features_train,
                                                           pos_train,
                                                           stanford_train,
                                                           lexicons_train,
                                                           data_test,
                                                           labels_test,
                                                           features_test,
                                                           pos_test,
                                                           stanford_test,
                                                           lexicons_test,
                                                           EPOCHS,
                                                           BATCH_SIZE,
                                                           SEED,
                                                           MIN_IMPROVEMENT,
                                                           IMPROVEMENT_PATIENCE,
                                                           floydhub,
                                                           auxOutputs,
                                                           saveWeights,
                                                           OPTIMIZER)

        log_to_json(EMBEDDING_DIM,
                    list_of_avg_recalls.index(max(list_of_avg_recalls)) + 1,
                    MAX_SEQUENCE_LENGTH,
                    MAX_NB_WORDS,
                    BATCH_SIZE,
                    SEED,
                    EMBEDDING_TOOL,
                    auxOutputs,
                    attention,
                    NUMBER_OF_FILTERS,
                    FILTER_KERNEL_SIZE,
                    POS_FILTER_KERNEL_SIZE,
                    MAX_POOLING_WINDOW,
                    max(list_of_avg_recalls),
                    list_of_recalls[list_of_avg_recalls.index(max(list_of_avg_recalls))],
                    NOISE_INPUT,
                    DROP_TEXT_INPUT,
                    DROP_CASTLE,
                    DROP_EMB_TOWER,
                    JSON_FILE,
                    OPTIMIZER,
                    L1,
                    L2,
                    less_data)


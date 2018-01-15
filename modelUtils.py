import keras
from keras.models import Model
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Flatten, GaussianNoise, \
    Conv1D, MaxPooling1D, Embedding, Reshape, Permute, LSTM, Bidirectional
from keras.utils import plot_model
from utils import *


def get_rnn(unit, tool, cells, bi=False, return_sequences=True, dropout=0.):
    rnn = unit(cells, return_sequences=return_sequences, dropout=dropout, name=tool + "_tower_LSTM")
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn


def create_attention_vector(input_, emb_mat_, emb_dim_, max_seq_len_, nb_filters_, filter_kernel, tool):
    # Implement Embedding Attention Vector of "Lexicon Integrated CNN Models with Attention for Sentiment Analysis"
    # paper (Bonggun et al., 2017)
    # n: number of words
    # m: number of filters
    # d: embedding dimention

    embedding_layer = Embedding(len(emb_mat_),
                                emb_dim_,
                                weights=[emb_mat_],
                                input_length=max_seq_len_,
                                trainable=False)

    document_matrix = embedding_layer(input_)  # dim: (n * d)

    cnn = Conv1D(nb_filters_, filter_kernel, activation='relu', name=tool + "_tower_covn1")(
        document_matrix)  # dim: (n * m)

    # trans = Reshape((int(cnn.shape[2]), int(cnn.shape[1])))(cnn)
    trans = Permute((2, 1))(cnn)  # elpizo oti einai san to transpose

    attention_vector = GlobalMaxPooling1D(name=tool + "_tower_globalmaxpooling")(trans)  # dim: (n)

    attention_vector_reshape = Reshape((int(attention_vector.shape[1]), 1))(attention_vector)

    # document_matrix_transpose = Reshape((int(document_matrix.shape[2]), int(document_matrix.shape[1])))(document_matrix)

    embedding_attention_vector = keras.layers.dot([document_matrix, attention_vector_reshape], axes=1)

    embedding_attention_vector = Flatten()(embedding_attention_vector)

    return embedding_attention_vector


def create_embedding_tower(input_, emb_mat_, emb_dim_, max_seq_len_, max_pool_win_, nb_filters_, filter_kernel_, noise_, drop_text_input_, drop_emb_tower_, attentionFlag_, tool):

    embedding_layer_glove = Embedding(len(emb_mat_),
                                emb_dim_,
                                weights=[emb_mat_],
                                input_length=max_seq_len_,
                                trainable=False)

    embedded_sequences = embedding_layer_glove(input_)
    tower = GaussianNoise(noise_)(embedded_sequences)
    tower = Dropout(drop_text_input_)(tower)
    tower = Conv1D(nb_filters_, filter_kernel_, activation='relu', name=tool + "_tower_covn1")(tower)
    tower = MaxPooling1D(max_pool_win_, name=tool + "_tower_maxpooling")(tower)
    tower = Conv1D(nb_filters_, filter_kernel_, activation='relu', name=tool + "_tower_covn2")(tower)
    tower = GlobalMaxPooling1D(name=tool + "_tower_globalmaxpooling")(tower)

    # Attention tower
    if attentionFlag_:
        attention_vec = create_attention_vector(input_, emb_mat_, emb_dim_, max_seq_len_, nb_filters_,
                                                      filter_kernel=1, tool="attention" + tool)
        tower = keras.layers.concatenate([tower, attention_vec],
                                          name="attention_concatenation" + tool)

    tower = Dropout(drop_emb_tower_, name=tool + "tower_dropout")(tower)

    return tower


def create_stanford_tower(input_):

    tower = Flatten()(input_)
    print(input_.shape[2])
    print(input_.shape[1])
    for i in range(int(input_.shape[2]) - 1):
        tower = Dense(int(((int(input_.shape[1])*int(input_.shape[2])) / int(input_.shape[2])) * (int(input_.shape[2]) - i)), activation='sigmoid', name='stanford_dense_' + str(i))(tower)

    tower = Dense(int(input_.shape[2]), activation='sigmoid', name='stanford_dense')(tower)

    return tower


def cnn_multi_filters(input_, emb_mat_, emb_dim_, max_seq_len_, nb_filters_, nfilters, attentionFlag_, tool):

    embedding_layer_glove = Embedding(len(emb_mat_),
                                emb_dim_,
                                weights=[emb_mat_],
                                input_length=max_seq_len_,
                                trainable=False)

    embedded_sequences = embedding_layer_glove(input_)

    pooling_reps = []
    for i in nfilters:
        feat_maps = Conv1D(nb_filters_, i, activation='relu', name=tool + "_1_conv-filters-" + str(i))(embedded_sequences)
        pool_vecs = MaxPooling1D(pool_length=2)(feat_maps)
        pool_vecs = Conv1D(nb_filters_, i, activation='relu', name=tool + "_2_conv-filters-" + str(i))(pool_vecs)
        pool_vecs = GlobalMaxPooling1D(name=tool + "globalmaxpooling-filters-" + str(i))(pool_vecs)

        pool_vecs = Dense(nb_filters_, activation='relu', name="dense" + str(i))(pool_vecs)
        # pool_vecs = Flatten()(pool_vecs)
        # pool_vecs = GlobalMaxPooling1D()(feat_maps)
        pooling_reps.append(pool_vecs)

    representation = keras.layers.concatenate(pooling_reps, name=tool + "concat")

    # Attention tower
    if attentionFlag_:
        attention_vec = create_attention_vector(input_, emb_mat_, emb_dim_, max_seq_len_, nb_filters_,
                                                      filter_kernel=1, tool="attention" + tool)
        representation = keras.layers.concatenate([representation, attention_vec],
                                          name="attention_concatenation" + tool)

    return representation


def create_pos_tower(input_, emb_mat_, emb_dim_, max_seq_len_, output_dim, tool="POS"):

    embedding_layer_glove = Embedding(len(emb_mat_),
                                emb_dim_,
                                weights=[emb_mat_],
                                input_length=max_seq_len_,
                                trainable=False)

    embedded_sequences = embedding_layer_glove(input_)

    lstm = get_rnn(unit=LSTM, cells=64, bi=True, tool=tool, dropout=0.1)

    tower = lstm(embedded_sequences)
    tower = Flatten()(tower)
    tower = Dense(output_dim, activation='relu', name=tool + "tower_LSTM")(tower)

    return tower


def create_model(**kwargs):

    emb_mat_glove = kwargs.get('emb_mat_glove', None)
    emb_mat_w2v = kwargs.get('emb_mat_w2v', None)
    emb_mat_pos = kwargs.get('emb_mat_pos', None)
    emb_pos_dim = kwargs.get('emb_pos_dim', None)
    emb_dim = kwargs.get('emb_dim', None)
    max_seq_len = kwargs.get('max_seq_len', None)
    max_pool_win = kwargs.get('max_pool_win', None)
    nb_filters = kwargs.get('nb_filters', None)
    filter_kernel = kwargs.get('filter_kernel', None)
    pos_filter_kernel = kwargs.get('pos_filter_kernel', None)
    out_dim = kwargs.get('out_dim', None)
    feature_len = kwargs.get('features_len', None)
    pos_len = kwargs.get('pos_len', None)
    noise = kwargs.get('noise', None)
    drop_text_input = kwargs.get('drop_text_input', None)
    drop_emb_tower = kwargs.get('drop_emb_tower', None)
    drop_castle = kwargs.get('drop_castle', None)
    stanford_shape = kwargs.get('stanford_shape', None)
    attentionFlag = kwargs.get('attentionFlag', None)
    auxOutputsFlag = kwargs.get('auxOutputsFlag', None)


    main_input = Input(shape=(max_seq_len,), dtype='int32', name="main_input")


    # Glove tower
    # glove_tower = create_embedding_tower(main_input, emb_mat_glove, emb_dim, max_seq_len, max_pool_win, nb_filters, filter_kernel, noise, drop_text_input, drop_emb_tower, attentionFlag, tool="glove")

    # Word2Vec tower
    w2v_tower = create_embedding_tower(main_input, emb_mat_w2v, emb_dim, max_seq_len, max_pool_win, nb_filters, filter_kernel, noise, drop_text_input, drop_emb_tower, attentionFlag, tool="w2v")

    # Features tower
    features_input = Input(shape=(feature_len,), dtype='float32', name="features_input")
    # tower_2 = Dense(7, activation='relu', name="tower_2")(features_input)
    # tower_2 = BatchNormalization()(features_input)

    # POS tower
    pos_input = Input(shape=(pos_len,), dtype='float32', name="POS_input")
    # pos_tower = create_embedding_tower(pos_input, emb_mat_pos, emb_pos_dim, max_seq_len, nb_filters, pos_filter_kernel, tool="POS")
    pos_tower = cnn_multi_filters(pos_input, emb_mat_pos, emb_pos_dim, max_seq_len, nb_filters, pos_filter_kernel, attentionFlag, tool="POS")
    # pos_tower = create_pos_tower(pos_input, emb_mat_pos, emb_pos_dim, max_seq_len, output_dim=50)  # LSTM

    # Stanford tower
    stanford_input = Input(shape=(stanford_shape[1], stanford_shape[2],), dtype='float32', name="stanford_input")
    stanford_tower = create_stanford_tower(stanford_input)

    # Auxiliary outputs
    # auxiliary_output_glove = Dense(out_dim, activation='sigmoid', name='aux_output_glove')(glove_tower)
    auxiliary_output_w2v = Dense(out_dim, activation='sigmoid', name='aux_output_w2v')(w2v_tower)
    auxiliary_output_pos = Dense(out_dim, activation='sigmoid', name='aux_output_pos')(pos_tower)

    # Merge
    castle = keras.layers.concatenate([w2v_tower, features_input, pos_tower, stanford_tower], name="castle_concatenation")
    # castle = Flatten()(castle)

    castle = Dropout(drop_castle, name="dropout_after_merge")(castle)
    castle = Dense(nb_filters, activation='relu', name="castle_dense")(castle)
    main_output = Dense(out_dim, activation='softmax', name="predictions")(castle)

    if auxOutputsFlag:
        model_ = Model(inputs=[main_input, features_input, pos_input, stanford_input], outputs=[main_output, auxiliary_output_w2v, auxiliary_output_pos])
    else:
        model_ = Model(inputs=[main_input, features_input, pos_input, stanford_input], outputs=main_output)

    # plot_model(model_, to_file="plots/pos_lstm.png")
    model_.summary()

    return model_


def train_model(model_, x_train_, y_train_, features_train_, pos_train_, stanford_train_,
                        x_test_, y_test_, features_test_, pos_test_, stanford_test_,
                        epochs, batch, seed, min_improvement, imp_patience,
                        floydhub_flag, aux_outputs_flag, save_weights_flag, optimizer):
    # train model
    model_.compile(loss='categorical_crossentropy',
                   optimizer=optimizer)

    epochAvgRecall = []
    recalls = []

    isImproved = []

    for epoch in range(epochs):
        if aux_outputs_flag:
            model_.fit([x_train_, features_train_, pos_train_, stanford_train_], [y_train_, y_train_, y_train_],
                       batch_size=batch,
                       epochs=1,
                       validation_split=0.05)
        else:
            model_.fit([x_train_, features_train_, pos_train_, stanford_train_], y_train_,
                       batch_size=batch,
                       epochs=1,
                       validation_split=0.05)

        [avg_recall, recall_p, recall_u, recall_n] = avg_recall_on_training_end(aux_outputs_flag, model_, x_test_, y_test_, features_test_, pos_test_, stanford_test_)
        print("Average recall on Epoch " + str(epoch + 1) + "/" + str(epochs) + ": " + str(avg_recall))

        # serialize weights to HDF5
        if save_weights_flag:
            save_weights(model_, seed, epoch, avg_recall, floydhub_flag)

        # Early stopping criteria
        if epoch == 0:
            epochAvgRecall.append(avg_recall)
            recalls.append([recall_p, recall_u, recall_n])
            isImproved.append(False)
        else:
            if avg_recall >= max(epochAvgRecall) + min_improvement:
                improvement = True
            else:
                improvement = False

            epochAvgRecall.append(avg_recall)
            recalls.append([recall_p, recall_u, recall_n])
            isImproved.append(improvement)


            if epoch >= imp_patience:
                if True in isImproved[epoch - imp_patience:epoch+1]:  # to check the current epoch also
                    continue
                else:
                    print("Early stopping criteria met")
                    print("Training stopped")
                    break

    print("Max avg_recall: %s, found in epoch %s" % (max(epochAvgRecall), epochAvgRecall.index(max(epochAvgRecall)) + 1))

    print("Model trained")

    return epochAvgRecall, recalls

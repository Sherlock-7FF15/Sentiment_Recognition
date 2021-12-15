import numpy as np
import pickle
import os
#
import utils as utl
# from gensim.models import Word2Vec


# create an index vec, something like
# [ 2,5,27,9,0,0....]
# adapt code so that if we are
def makeIndexVec(words, model, maxLength=50, withZeros=False, padLeft=True):
    indexVec = np.zeros((maxLength,), dtype="float32")

    vocab = model.wv.vocab
    index2word_set = set(model.wv.index2word)

    if padLeft:
        # we shall always choose maxLength now.
        i = maxLength - 1
        for word in reversed(words):
            if i == 0:
                break
            if word in index2word_set:
                indexVec[i] = vocab[word].index
            i = i - 1
    else:
        i = 0
        for word in words:
            if i == maxLength:
                break
            if word in index2word_set:
                indexVec[i] = vocab[word].index
            i = i + 1

    return indexVec


def indices_to_words(indices, model, maxLength=50, padLeft = True):
    wordVec = [""] * maxLength
    index2word = model.index2word

    if padLeft:
        # we shall always choose maxLength now.
        i = maxLength - 1
        for index in reversed(indices):
            if i == 0:
                break

            if i < 0:
                wordVec[i] = ""
            else:
                wordVec[i] = index2word[index]
            i = i - 1
    else:
        i = 0
        for index in indices:
            if i == maxLength:
                break

            if i < 0:
                wordVec[i] = ""
            else:
                wordVec[i] = index2word[index]
            i = i + 1

    return wordVec


# 3.
def get_indices_word2vec(data, column, model, maxLength=50, writeIndexFileName="./model/word2vec_indices.pickle",
                         padLeft=True, keep_freqwords=[]):

    if (os.path.isfile(writeIndexFileName)):
        reviewIndexVecs = pickle.load(open(writeIndexFileName, 'rb'))
        return reviewIndexVecs
    #
    messages = data[column]
    messages = np.array([utl.preprocess_ST_message(message) for message in messages])
    texts = []
    for i in messages:
        texts.append(i.split())
        # texts.append(text)
    # print('4544',texts)
    # reviews = read_article.data_to_reviews(data, column, keep_freqwords=keep_freqwords)
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewIndexVecs = np.zeros((len(texts), maxLength), dtype="int32")
    #
    # Loop through the reviews
    for review in texts:
        # Print a status message every 1000th review
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(texts)))
        # Call the function (defined above) that makes average feature vectors
        reviewIndexVecs[counter] = makeIndexVec(review, model, maxLength, padLeft=padLeft)
        # Increment the counter
        counter = counter + 1

    pickle.dump(reviewIndexVecs, open(writeIndexFileName, 'wb'))
    return reviewIndexVecs


# 2.
def create_embedding_weights(model, max_index=0, writeEmbeddingFileName = "./model/embedding_weights.pkl"):

    if (os.path.isfile(writeEmbeddingFileName)):
        reviewIndexVecs = pickle.load(open(writeEmbeddingFileName, 'rb'))
        return reviewIndexVecs

    # dimensionality of your word.txt vectors
    num_features = len(model[list(model.wv.vocab.keys())[0]])  # 300
    print('486548648',model.wv.vocab.keys())
    print('486548648',list(model.wv.vocab.keys()))
    print('486548648',list(model.wv.vocab.keys())[0])
    print('486548648',model[list(model.wv.vocab.keys())[0]])
    print('486548648',len(model.wv.vocab))

    n_symbols = len(model.wv.vocab) + 1  # adding 1 to account for 0th index (for masking)

    # Only word2vec feature set
    embedding_weights = np.zeros((max(n_symbols + 1, max_index + 1), num_features))
    for word, value in model.wv.vocab.items():
        print(embedding_weights.shape,word,value,value.index,model[word])
        embedding_weights[value.index, :] = model[word]
        print(embedding_weights.shape)
        break

    pickle.dump(embedding_weights, open(writeEmbeddingFileName, 'wb'))

    return embedding_weights


# 1.data_labeled, "review", num_features=300,downsampling=1e-3,model_path=self.word2vecmodel_path
def get_word2vec_model(data, column, num_features=300, min_word_count=10, num_workers=4, context=10,
                           downsampling=1e-3,
                           model_path="./model/300features_40minwords_10context", remove_stopwords=False):

    if os.path.isfile(model_path):
        return load_word2vec_model(model_path)

    # Set values for various parameters
    # num_features = 300    # Word vector dimensionality
    # min_word_count = 10   # Minimum word.txt count
    # num_workers = 4       # Number of threads to run in parallel
    # context = 10          # Context window size
    # downsampling = 1e-3   # Downsample setting for frequent words

    # sentences = read_article.data_to_sentences(data, column, remove_stopwords=remove_stopwords)
    messages = data[column]
    messages = np.array([utl.preprocess_ST_message(message) for message in messages])
    texts = []
    for i in messages[:2000]:
        texts.append(i.split())
        # texts.append(text)
    # print('4544',texts)
    for i in range(2):
        print("Sentiment: {}".format(texts[i]))

    # Initialize and train the model (this will take some time -- we are using everything here)
    print("Training model...")
    model = Word2Vec(texts, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling)
    # train_vecs = np.concatenate([build_sentence_vector(z, num_features, model) for z in x_train])
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_path)

    return model


def load_word2vec_model(model_name="300features_40minwords_10context"):
    model = Word2Vec.load(model_name)
    return model


# Function to produce average
# for all of the word.txt vectors in a given paragraph
# This code may be refactors later...
def makeAvgVec(words, model):
    #
    # Pre-initialize an empty numpy array (for speed)
    #
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    num_features = len(model[model.vocab.keys()[0]])
    #
    # Loop over each word.txt in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    # This is if we are running sentence level
    featureVec = np.zeros((num_features,), dtype="float32")
    i = 0
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])


    if nwords > 0:
        # return np.average(featureVec, axis = 0)
        featureVec = np.divide(featureVec, nwords)

    return featureVec


# Given a set of reviews (each one a list of words), calculate
# the average feature vector for each one and return a 2D numpy array
# This is pretty much lifted from the Kaggle tutorial.
def get_avgfeatures_word2vec(data, column, model, num_features=300, writeFeaturesFileName = "./model/imdb_avgfeatures.pickle" ):
    if(os.path.isfile(writeFeaturesFileName)):
        reviewFeatureVecs = pickle.load(open(writeFeaturesFileName))
        return reviewFeatureVecs
    #
    reviews = read_article.data_to_reviews(data, column)
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        # Call the function (defined above) that makes average feature vectors

        reviewFeatureVecs[counter] = makeAvgVec(review, model)
        # Increment the counter
        counter = counter + 1

    pickle.dump(reviewFeatureVecs, open(writeFeaturesFileName, 'w'))
    return reviewFeatureVecs





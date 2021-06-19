import argparse
import csv
import gensim
import os
import pathlib
import pickle
import random
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import reuters
from scipy import spatial

# global variables
stop_words = set(stopwords.words('english'))
Pkl_Filename = "Pickle_Model.pkl"


# For custom datasets
def parse_dataset(data_folder_name):
    try:
        lst_of_docs = []
        data_dir = str(pathlib.Path().absolute()) + data_folder_name

        for folder in os.listdir(data_dir):
            sub_path = "".join((data_dir, "/", folder))  # Update path
            files = os.listdir(sub_path)  # Get files
            for file in files:
                sub_path = "".join((data_dir, "/", folder, "/", file))  # Update path
                with open(sub_path) as f:  # Open file(s)
                    lst_of_docs.append(f.read())
        return lst_of_docs
    except ValueError:
        print('Error: Unable to parse dataset!')
        quit()


# Clean the data
def tokenize(X_raw):
    X = []
    for doc in X_raw:
        words = word_tokenize(doc)
        words_filtered = [word for word in words if word not in stop_words]
        X.append(words_filtered)
    return X


# Covert to gensim format
def tagged_document(list_of_list_of_words):
    print('--> Converting to gensim format...')  # For status updates
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def write_to_csv(input, output_file, delimiter='\t'):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(input)


def run(args):
    mode = args.mode
    X_raw, X = [], []

    if mode == "train":
        print('--> Fetching data...')  # For status updates
        if not args.training_data:  # Default training data
            X_raw = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data
        elif args.training_data == 'reuters':
            documents = reuters.fileids()
            train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
            X_raw = [reuters.raw(doc_id) for doc_id in train_docs_id]
        elif args.training_data == 'custom':
            X_raw = parse_dataset('/custom/')
        else:
            print(f"{args.training_data} is incompatible.")
            quit()

        print('--> Tokenizing...')
        X = tokenize(X_raw)  # Clean the data
        train_corpus = list(tagged_document(X))  # Call to convert

        print('--> Creating the model...')  # For status updates
        model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        # Save the model
        print(f'--> Saving the model as {Pkl_Filename}...')  # For status updates
        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(model, file)

        print('***Training complete!***')  # For status updates

    elif mode == "run":
        # Load the trained model
        print('--> Loading saved model...')  # For status updates
        with open(Pkl_Filename, 'rb') as f:
            model = pickle.load(f)

        if args.document:
            # Load the document we want to compare
            print('--> Grabbing the document...')  # For status updates
            with open(args.document) as f:
                X_raw.append(f.read())
        elif args.query:
            print('--> Grabbing the query...')
            X_raw = args.query

        # Stop program if no values passed
        else:
            print("Error: Please submit a query or a document via --document or --query.")
            quit()
        if not args.data:
            print('Error: please pass the name of the folder with data/documents in it')
            quit()

        print('--> Loading data...')
        tmp_X_raw = parse_dataset('/'+args.data+'/')  # Grab the unknown documents
        X_raw = X_raw + tmp_X_raw  # Merge our file with the other documents
        print('--> Tokenizing...')
        X = tokenize(X_raw)  # Clean the data

        print('--> labeling documents...')
        doc_tracker = {}
        for num, doc in enumerate(X):
            doc_tracker['Document ' + str(num)] = doc

        print('--> Defining the relations between documents...')
        veclist_metadata, veclist = [], []
        for num in doc_tracker.keys():
            tmp = model.infer_vector(doc_tracker[num])
            veclist.append(list(tmp))
            veclist_metadata.append([num])

        print('--> Writing results to files...')
        # http://projector.tensorflow.org/
        write_to_csv(veclist, "vectors.csv")
        write_to_csv(veclist_metadata, "vectors_metadata.csv")

        top_ten_values, top_ten_doc_ids = [0] * 10, ['None'] * 10
        our_vec = model.infer_vector(X[0])

        print('--> Finding the ten closest documents to our query/document...')
        for i, doc in enumerate(X[1:]):
            next_vec = model.infer_vector(doc)
            cos_distance = spatial.distance.cosine(our_vec, next_vec)
            for j in range(len(top_ten_values)):
                if cos_distance > top_ten_values[j]:
                    top_ten_values.insert(j, cos_distance)
                    top_ten_doc_ids.insert(j, f'Document {i + 1}')
                    top_ten_values.pop()
                    top_ten_doc_ids.pop()
                    break

        print('--> Writing closest documents to CSV files...')
        tmp_dict = {'Doc ID': top_ten_doc_ids, 'Values': top_ten_values}
        df = pd.DataFrame(tmp_dict)
        df.to_csv("run.csv", encoding='utf-8', index=False)
        print(df)

        with open('run_text.txt', 'w') as f:
            for key in top_ten_doc_ids:
                f.write(key + '\n' + str(doc_tracker[key]) + '\n\n')

        print('\n***Run complete!***\n')

    elif mode == "eval":
        print('--> Loading saved model...')  # For status updates
        with open(Pkl_Filename, 'rb') as f:
            model = pickle.load(f)

        category_X_dict, categories, X = {}, [], []
        if not args.test_data:
            print('--> Fetching the test data...')  # For status updates
            # probably a better way to pull this
            newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            categories = newsgroups_test.target_names
            print('--> Tokenizing...')
            for cat in categories:
                X_raw = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                           categories=[cat]).data
                X = tokenize(X_raw)
                category_X_dict[cat] = X

        elif args.test_data == 'reuters':
            print('--> Fetching the test data...')  # For status updates
            categories = reuters.categories()
            category_X_dict_raw = {}
            for cat in categories:
                docs = reuters.fileids(cat)
                test_docs_id = list(filter(lambda doc: doc.startswith("test"), docs))
                category_X_dict_raw[cat] = test_docs_id
            print('--> Tokenizing...')
            for cat in category_X_dict_raw.keys():
                for doc in category_X_dict_raw[cat]:
                    X_raw = reuters.raw(doc)
                    tmp_X = word_tokenize(X_raw)
                    X = [word for word in tmp_X if word not in stop_words]
                    if cat in category_X_dict.keys():
                        X = " ".join(X)
                        category_X_dict[cat].append([X])
                    else:
                        X = " ".join(X)
                        category_X_dict[cat] = [[X]]
        else:
            print(f'Argument {args.test_data} invalid.')
            quit()

        # There is probably a better way to store/organize this...
        category_num_ids = {ID: cat for ID, cat in enumerate(categories)}  # Giving categories numeric ID's
        # Paring categories against one another i.e. (1,2), (10, 5), etc.
        category_pairs_dict = {tuple(sorted([ID, id2])): [] for ID in category_num_ids for id2 in category_num_ids}
        total_dict = {'Matching': 0, "Non-Matching": 0}
        matching_num, non_matching_num = 0, 0

        print('--> Evaluating model...')
        for j, pair in enumerate(category_pairs_dict):
            category_pairs_dict[pair] = 0  # Initialize document pair score to zero
            names = (category_num_ids[pair[0]], category_num_ids[pair[1]])
            print(f"    > Comparing categories '{names[0]}' and '{names[1]}'")
            for i in range(100):
                # Set to grab random documents...
                random_index_1 = random.randint(0, len(category_X_dict[names[0]]) - 1)
                random_index_2 = random.randint(0, len(category_X_dict[names[1]]) - 1)

                # Grab random document
                doc1, doc2 = category_X_dict[names[0]][random_index_1], category_X_dict[names[1]][random_index_2]
                res1, res2 = model.infer_vector(doc1), model.infer_vector(doc2)  # Get scores
                cos_distance = spatial.distance.cosine(res1, res2)

                # Add the score
                if category_pairs_dict[pair] != 0:
                    category_pairs_dict[pair] = (category_pairs_dict[pair] + cos_distance) / i  # mean/average
                else:
                    category_pairs_dict[pair] = cos_distance

            if names[0] == names[1]:  # Could check the nums but...
                total_dict["Matching"] += category_pairs_dict[pair]  # same thing as above, but we will divide later
                matching_num += 1
            else:
                total_dict["Non-Matching"] += category_pairs_dict[pair]
                non_matching_num += 1

        print('--> Making final calculations...')
        # Now find the mean/average
        total_dict["Matching"] = total_dict["Matching"] / matching_num
        total_dict["Non-Matching"] = total_dict["Non-Matching"] / non_matching_num

        print(f'--> Number of matching categories {matching_num}, number of non-matching categories {non_matching_num}')

        df_all = pd.DataFrame(list(total_dict.items()),
                              columns=['Groups', 'Cosine Distance'])
        df_individual = pd.DataFrame(list(category_pairs_dict.items()),
                                     columns=['Pairs', 'Cosine Distance'])

        print('--> Writing results to CSV files...')
        df_all.to_csv("all.csv", encoding='utf-8', index=False)
        df_individual.to_csv("pairs.csv", encoding='utf-8', index=False)

        print('\n***Eval complete!***\n')
        print(df_all, '\n')
        print(df_individual)
    else:
        print(f"{args.mode} is an incompatible mode.")
        quit()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="")
    PARSER.add_argument('--mode', metavar='M', type=str, help="A document retrieval program utilizing a Doc2Vec model.")
    PARSER.add_argument('--training_data', metavar='X', type=str, help="Chose the training set of this model: "
                                                                       "Newsgroup (Default), Reuters, or Custom")
    PARSER.add_argument('--test_data', metavar='Y', type=str, help="Chose the test set of this model: Newsgroup ("
                                                                   "Default) or Reuters.")
    PARSER.add_argument('--data',  metavar='F', type=str, help="Name of the folder where the unknown documents are "
                                                               "stored.")
    PARSER.add_argument('--document', metavar='D', type=str, help="Give a document in order to find similar documents.")
    PARSER.add_argument('--query', metavar='Q', type=str, nargs='+', help="Give a query in order to find similar "
                                                                          "documents.")
    ARGS = PARSER.parse_args()
    run(ARGS)

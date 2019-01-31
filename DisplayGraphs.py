
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sentiment_strings = {
    0: "negative",
    1: "somewhat negative",
    2: "neutral",
    3: "somewhat positive",
    4: "positive"
}

def displayGraph(filename_train):
    # Read train CSV.
    train_data = pd.read_csv(filename_train, sep='\t')

    # Count sentiments
    sentiments_count = [(train_data.iloc[:, 3] == key).sum() for key in sentiment_strings]

    # Calculate the sum.
    total_count = sum(sentiments_count)

    # Count unique SentenceID.
    sentenceIDs = np.array(train_data["SentenceId"])
    sentenceID_count = len(set(list(sentenceIDs)))

    # Count skipped sentenceIDs
    last = 0
    indexed_skipped = []
    for s in sentenceIDs:
        if s != last and s != last + 1:
            indexed_skipped.append(s)
        last = s

    # Calculating percentages of each sentiment.
    percentages = []
    for number in sentiments_count:
        percentages.append(str(round(number * 100 / total_count, 3)) + "%")

    # Creating the dataframe.
    data = pd.DataFrame()
    data["Sentiments"] = sentiment_strings.values()
    data["Count"] = sentiments_count
    data["Percentages"] = percentages

    # print info.
    print(data)
    print("")
    print("phrase_count = " + str(total_count))
    print("sentenceID_count = " + str(sentenceID_count))
    print("sentenceID_skipped = " + str(indexed_skipped))

    # Show dataframe with seaborn.
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Sentiments", y="Count", data=data)
    plt.show()

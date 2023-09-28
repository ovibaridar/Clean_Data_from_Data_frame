import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path
path1 = "File/country-list.csv"
path2 = "File/university_of_bangladesh.csv"
path3 = "File/bangladeash_collage.xlsx"
path4 = "File/que.csv"

# DataFrame
capitals = pd.read_csv(path1)
bangladesh_universities = pd.read_csv(path2)
bangladesh_Collage = pd.read_excel(path3)
reg_que = pd.read_csv(path4)

while True:

    # User input
    user_input = input("User: ")
    if user_input == "*":
        break

    # Define a list of data frames
    data_frames = [capitals, bangladesh_universities, bangladesh_Collage, reg_que]

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Initialize variables to keep track of similarity scores and responses
    max_similarity = -1.0
    best_responses = []

    # Set a threshold for similarity
    threshold = 0.5  # You can adjust this threshold as needed

    # Iterate through the data frames and check for similarity
    for df in data_frames:
        if 'que' in df.columns and 'ans' in df.columns:
            # Fit and transform the questions in the data frame
            tfidf_matrix = vectorizer.fit_transform(df['que'])

            # Transform the user's input
            user_tfidf = vectorizer.transform([user_input])

            # Calculate cosine similarities between user input and dictionary questions
            cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)

            # Find the index of the most similar question
            most_similar_index = cosine_similarities.argmax()

            # Get the corresponding answer and similarity score
            similarity = cosine_similarities[0][most_similar_index]

            # If the similarity is above the threshold, consider it a match
            if similarity > threshold:
                best_responses.append((df.columns, df.iloc[most_similar_index]))

    # Print the matching rows with column names from all data frames
    if best_responses:
        print("Matching Responses:")
        for columns, response in best_responses:
            print("Matching Columns:", columns)
            print("Matching Row:")
            print(response)
    else:
        print("No matching answer found.")

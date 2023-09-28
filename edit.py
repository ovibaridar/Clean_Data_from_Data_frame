import pandas as pd

# Path
path2 = "File/university_of_bangladesh.csv"

# DataFrame
capitals = pd.read_csv(path2)
# Also division is
capitals["Ph.D. granting"] = "Ph.D. granting " + capitals["Ph.D. granting"] + " ."

print(capitals["Ph.D. granting"])

capitals.to_csv(path2, index=False)

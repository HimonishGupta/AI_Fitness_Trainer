import pandas as pd

files = [
    "pose_data_squat.csv",
    "pose_data_pushup.csv",
    "pose_data_bicep_curl.csv",
    "pose_data_tricep_extension.csv",
    "pose_data_shoulder_press.csv"
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)
df.to_csv("pose_data.csv", index=False)

print(" All data combined and saved to pose_data.csv")

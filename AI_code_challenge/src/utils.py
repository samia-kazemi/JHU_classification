import random

def add_data_split(df, sp_label_col = 'y', id_col = 'SpecimenID', ratio = [0.70, 0.15, 0.15]):
    specimenID_to_split = {}
    for y in range(max(set(df[sp_label_col])) + 1):
        sub_df = df[df['y'] == y]
        unique_SpecimenID = random.sample(list(set(sub_df[id_col])), len(list(set(sub_df[id_col]))))
        split_count = [round(ratio[0] * len(unique_SpecimenID)),
                       round((ratio[0] + ratio[1]) * len(unique_SpecimenID))]
        SpecimenID_split = [unique_SpecimenID[:split_count[0]]]       # Specimens from Train set
        SpecimenID_split.append(unique_SpecimenID[split_count[0]:split_count[1]])    # Specimens from Valid set
        SpecimenID_split.append(unique_SpecimenID[split_count[1]:])   # Specimens from Test set
        for id in SpecimenID_split[0]:
            specimenID_to_split[id] = "Train"
        for id_val in SpecimenID_split[1]:
            specimenID_to_split[id_val] = 'Valid'
        for id_ts in SpecimenID_split[2]:
            specimenID_to_split[id_ts] = 'Test'
    df['Split'] = df[id_col].map(specimenID_to_split)
    for y in range(max(set(df[sp_label_col])) + 1):
        sub_df = df[df['y'] == y]
        for split in ['Train', 'Valid', 'Test']:
            subsub_df = sub_df[sub_df['Split'] == split]
            print(f"For y = {y}, no. of unique specimens in {split} = {len(set(subsub_df[id_col]))}")

    return df
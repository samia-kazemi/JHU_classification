import random
import matplotlib.pyplot as plt
import seaborn as sns

def add_data_split(df, sp_label_col = 'y', id_col = 'SpecimenID', target_ratio = [0.70, 0.15, 0.15]):
    specimenID_to_split = {}
    for y in range(max(set(df[sp_label_col])) + 1):
        sub_df = df[df['y'] == y]
        for folder in set(sub_df['ImageFolder']):
            subsub_df = sub_df[sub_df['ImageFolder'] == folder]
            unique_SpecimenID = random.sample(list(set(subsub_df[id_col])), len(list(set(subsub_df[id_col]))))
            if folder == 'kenya_01':
                split_count = [round(target_ratio[0] * len(unique_SpecimenID)),
                            round((target_ratio[0] + target_ratio[1]) * len(unique_SpecimenID))]
            else:
                split_count = [round(0.8 * len(unique_SpecimenID)), len(unique_SpecimenID)]
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
        for folder in set(sub_df['ImageFolder']):
            subsub_df = sub_df[sub_df['ImageFolder'] == folder]
            for split in ['Train', 'Valid', 'Test']:
                subsubsub_df = subsub_df[subsub_df['Split'] == split]
                print(f"For y = {y}, folder {folder}, no. of unique specimens in {split} = {len(set(subsubsub_df[id_col]))}")

    return df

def save_cm(cm, labels, save_dir):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # 4. Save the image to disk
    plt.savefig(save_dir /'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
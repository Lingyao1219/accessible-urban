import os
import pycountry
from tqdm import tqdm
import pandas as pd
import argparse
from words_list import include_list, exclude_list

include_pattern = '|'.join(include_list)
exclude_pattern = '|'.join(exclude_list)


def create_folder(folder_name):
    """Create a folder if it does not exist"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def filter_data(filename1, filename2):
    """Filter the review data based on the word list"""
    # Filter dataframe 2
    print(f"----- Filter file {filename2} -----")
    chunksize = 10000
    filtered_df2_chunks = []
    for chunk in tqdm(pd.read_json(filename2, compression='gzip', orient='records', lines=True, chunksize=chunksize),
                      desc='Processing chunks',
                      unit='chunk'):
        chunk['text'] = chunk['text'].astype(str)
        filtered_chunk = chunk[chunk['text'].str.contains(include_pattern, na=False) & 
                               ~chunk['text'].str.contains(exclude_pattern, na=False)]
        filtered_df2_chunks.append(filtered_chunk)

    filtered_df2 = pd.concat(filtered_df2_chunks)
    print(f"Filter file {filename2} ----- after: {len(filtered_df2)}")
    
    filtered_df2['gmap_id'] = filtered_df2['gmap_id'].astype(str)
    gmap_list = list(set(filtered_df2['gmap_id'].tolist()))

    # Filter dataframe 1
    df1 = pd.read_json(filename1, compression='gzip', orient='records', lines=True)
    df1['gmap_id'] = df1['gmap_id'].astype(str)
    filtered_df1 = df1[df1['gmap_id'].isin(gmap_list)]
    print(f"Filter file {filename1} ----- before: {len(df1)} after: {len(filtered_df1)}")    
    
    return filtered_df1, filtered_df2


def write_data(filtered_df1, filename1, filtered_df2, filename2, meta_folder, review_folder):
    """Write the filtered data to a new file"""
    write_filename1 = filename1.replace('meta', 'filtered_meta').split('.json')[0] + '.json'
    write_filename1 = os.path.join(meta_folder, write_filename1)
    write_filename2 = filename2.replace('review', 'filtered_review').split('.json')[0] + '.json'
    write_filename2 = os.path.join(review_folder, write_filename2)
    
    filtered_df1.to_json(write_filename1, orient='records', lines=True)
    print(f"Finished writing file: {write_filename1}")
    
    filtered_df2.to_json(write_filename2, orient='records', lines=True)
    print(f"Finished writing file: {write_filename2}\n")


def main(read_folder, project_name):
    """Main function to filter the data"""
    create_folder(f'{project_name}-meta')
    create_folder(f'{project_name}-review')
    
    countries = pycountry.countries
    us_states = [state.name.replace(" ", "_") 
                 if " " in state.name else state.name for state in pycountry.subdivisions.get(country_code='US')]

    filenames = os.listdir(read_folder)
    meta_files = sorted([filename for filename in filenames if filename.lower().startswith('meta')])
    review_files = sorted([filename for filename in filenames if filename.lower().startswith('review')])

    for state in us_states:
        for (filename1, filename2) in zip(meta_files, review_files):
            if (state.lower() in filename1.lower()) and (state.lower() in filename2.lower()):
                read_filename1 = os.path.join(read_folder, filename1)
                read_filename2 = os.path.join(read_folder, filename2)
                filtered_df1, filtered_df2 = filter_data(read_filename1, read_filename2)
                write_data(filtered_df1, filename1, filtered_df2, filename2, f'{project_name}-meta', f'{project_name}-review')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Google Map data")
    parser.add_argument('read_folder', type=str, help='Folder containing the data files')
    parser.add_argument('project_name', type=str, help='Base project name for the folders')
    args = parser.parse_args()
    main(args.read_folder, args.project_name)

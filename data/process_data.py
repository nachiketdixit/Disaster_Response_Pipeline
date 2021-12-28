import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    message_filepath - a string representing file path of messages.csv
    categories_filepath - a string representing file path of categories.csv
    
    OUTPUT
    df - a pandas dataframe resulted from 2 data source
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on=['id'])
    
    return df


def clean_data(df):
    '''
    INPUT
    df - a pandas dataframe resulted from 2 data source
    
    OUTPUT
    df - a pandas dataframe after cleaning and transforming
    '''
    categories = pd.Series(df['categories']).str.split(';', expand=True)
    row = categories.iloc[0]
    # Creating a dataframe of the 36 individual category columns
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # Converting category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    
    df.drop(columns = ['categories'], inplace = True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(keep='first', inplace=True)
    df.related.replace(2, 1, inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    INPUT
    df - a pandas dataframe
    database_filepath - the file path where we want to save dataframe result 
    
    OUTPUT
    No output, the dataframe is saved to data_filepath
    '''  
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
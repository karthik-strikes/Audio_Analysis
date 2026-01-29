#!/usr/bin/env python3
"""
Database Operations Module

Handles all database interactions including:
- Connection management
- Data insertion
- Feature table operations
- Processing status tracking
"""

import mysql.connector
from mysql.connector import IntegrityError
import pandas as pd
from typing import Optional
from config import Config


def connect_to_db():
    """
    Establish connection to MySQL database using configuration.
    
    Returns:
        mysql.connector.connection: Database connection object
        
    Raises:
        mysql.connector.Error: If connection fails
    """
    return mysql.connector.connect(**Config.get_db_config())


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful
    """
    try:
        conn = connect_to_db()
        conn.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def is_file_processed(filename: str, table_name: str, connection) -> bool:
    """
    Check if a file has already been processed and stored in database.
    
    Args:
        filename (str): Filename to check
        table_name (str): Database table name
        connection: MySQL connection object
        
    Returns:
        bool: True if already processed
    """
    query = f"SELECT COUNT(*) FROM {table_name} WHERE filename = %s"
    cursor = connection.cursor()
    cursor.execute(query, (filename,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] > 0


def insert_data(df: pd.DataFrame, table_name: str):
    """
    Insert main dataset into the specified database table.
    
    Inserts records with: message_id, message, KlaatchID, Date, CEL_Total, 
    CELVAL1, CELVAL2, CELVAL3, and Age columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to insert
        table_name (str): Name of the database table (e.g., 'merged_data')
    """
    connection = connect_to_db()
    cursor = connection.cursor()

    insert_query = f"""
        INSERT IGNORE INTO {table_name} (
            message_id, message, KlaatchID, Date, CEL_Total, CELVAL1, CELVAL2, CELVAL3, 
            Age
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Convert DataFrame to list of tuples with required columns
    required_cols = ['Filename', 'Text', 'KlaatchID', 'Date', 'CEL Total', 
                     'CELVAL1', 'CELVAL2', 'CELVAL3', 'Age']
    
    # Check if all columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, cannot insert data")
        cursor.close()
        connection.close()
        return
    
    data = df[required_cols].values.tolist()
    
    try:
        cursor.executemany(insert_query, data)
        connection.commit()
        print(f"✓ Inserted {cursor.rowcount} rows into {table_name}")
    except mysql.connector.Error as err:
        print(f"✗ Error: {err}")
    finally:
        cursor.close()
        connection.close()


def insert_feature_table_into_db(connection, dataframe: pd.DataFrame, table_name: str):
    """
    Insert feature dataframe into MySQL database.
    
    Args:
        connection: MySQL connection object
        dataframe (pd.DataFrame): Data to insert
        table_name (str): Target table name
    """
    cursor = connection.cursor()
    
    columns = ', '.join(dataframe.columns)
    placeholders = ', '.join(['%s'] * len(dataframe.columns))
    
    insert_query = f"""
        INSERT INTO {table_name} ({columns}) 
        VALUES ({placeholders})
    """
    
    inserted_count = 0
    
    for _, row in dataframe.iterrows():
        try:
            cursor.execute(insert_query, tuple(row))
            inserted_count += 1
        except IntegrityError:
            continue
    
    connection.commit()
    cursor.close()
    
    print(f"✓ Inserted {inserted_count}/{len(dataframe)} rows into {table_name}")


def retrieve_features_from_db(query: str) -> pd.DataFrame:
    """
    Retrieve features from database using custom query.
    
    Args:
        query (str): SQL query string
        
    Returns:
        pd.DataFrame: Query results as DataFrame
    """
    connection = connect_to_db()
    df = pd.read_sql(query, connection)
    connection.close()
    return df


def is_file_processed_opensmile(filename: str, connection) -> bool:
    """Check if file processed with OpenSmile."""
    return is_file_processed(filename, 'new_audio_features', connection)


def is_file_processed_librosa(filename: str, connection) -> bool:
    """Check if file processed with Librosa."""
    return is_file_processed(filename, 'new_librosa_features', connection)

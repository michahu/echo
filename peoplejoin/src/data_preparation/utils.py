import re


def find_tables_in_sql_query(sql_query) -> set[str]:
    # Regular expression pattern to match table names after FROM or JOIN
    pattern = r'(FROM|JOIN)\s+([`"]?(\w+)[`"]?)'

    # Find all matches for the pattern in the SQL query
    matches = re.findall(pattern, sql_query, re.IGNORECASE)

    # Extract the table names (third group in the pattern)
    tables = [match[2] for match in matches]

    # Return unique table names
    return set(tables)


def normalize_table_name(table_name: str) -> str:
    return table_name.lower().strip().replace("_", " ")


if __name__ == "__main__":
    # Example usage
    sql_query = """
    SELECT a.id, b.name
    FROM users a
    JOIN orders b ON a.id = b.user_id
    WHERE b.status = 'completed'
    """

    tables = find_tables_in_sql_query(sql_query)
    print(tables)

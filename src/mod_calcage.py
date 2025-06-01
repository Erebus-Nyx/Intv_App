# mod_calcage.py
"""
Python equivalent of Mod_CalcAge.bas
Contains the CalculateAge function.
"""
from datetime import datetime

def calculate_age(date_str):
    """Calculate age from a date string (YYYY-MM-DD or similar)."""
    try:
        birth_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            birth_date = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            return "Invalid date"
    today = datetime.today()
    years = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1
    if years > 0:
        return f"{years} year{'s' if years != 1 else ''}"
    else:
        months = (today.year - birth_date.year) * 12 + today.month - birth_date.month
        if today.day < birth_date.day:
            months -= 1
        if months > 0:
            return f"{months} month{'s' if months != 1 else ''}"
        else:
            days = (today - birth_date).days
            return f"{days} day{'s' if days != 1 else ''}"

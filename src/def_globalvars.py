# def_globalvars.py
"""
Python equivalent of Def_GlobalVars.bas
Contains global variables, constants, and array initializations.
"""

# Global variables (use None or suitable Python types)
ActiveRecordID = None
ActiveCaseNo = None
IDLookup = ""
CompleteNarrative = None
IntroNarrative = None
HomeAssessNarrative = None
AdultNarrative = None
AdultInterview = ""
ARInterview = None
CollateralNarrative = None
CollateralInterview = None
LeftTab = 0
TopTab = 0
PassVar = None
AdultFirst = None
ChildFirst = None

# Arrays (using lists for 1-based index compatibility)
Allegation = [[None for _ in range(6)] for _ in range(11)]  # 1 to 10, 1 to 6
Abbrev = [None] * 85  # 1 to 84
AdultArray = [[None for _ in range(5)] for _ in range(6)]  # 1 to 5, 1 to 5
ChildArray = [[None for _ in range(5)] for _ in range(6)]  # 1 to 5, 1 to 5
AR_Counts = [[None for _ in range(2)] for _ in range(2)]  # 1 to 2, 1 to 2
AR_Adult = [[None for _ in range(2)] for _ in range(9)]  # 1 to 8, 1 to 2
AR_AdultPres = []
AR_AdultHome = []
AR_Child = [[None for _ in range(2)] for _ in range(9)]  # 1 to 8, 1 to 2
AR_ChildPres = []
AR_ChildHome = []
Test = [[None for _ in range(8)] for _ in range(8)]  # 1 to 8, 1 to 8

# Functions to initialize arrays (from SetAdultArray, SetChildArray, DefineConstants)
def set_adult_array():
    for i in range(1, 6):
        AdultArray[i][0] = f"txt_FamilyName{i}"
        AdultArray[i][1] = f"txt_FamilyRelation{i}"
        AdultArray[i][2] = f"txt_FamilyDOB{i}"
        AdultArray[i][3] = f"txt_FamilySSN{i}"
        AdultArray[i][4] = f"txt_FamilyPhone{i}"

def set_child_array():
    for i in range(1, 6):
        ChildArray[i][0] = f"txt_ChildName{i}"
        ChildArray[i][1] = f"txt_ChildAge{i}"
        ChildArray[i][2] = f"txt_ChildDOB{i}"
        ChildArray[i][3] = f"txt_ChildSSN{i}"
        ChildArray[i][4] = f"txt_ChildBorn{i}"

def define_constants():
    Test[0][0] = "txt_AdultName1"
    set_child_array()
    set_adult_array()
    # Abbrev values (1-based)
    abbrevs = [
        "AB", "AD", "AG", "AN", "AT", "IN", "IP", "LA", "MO", "AU", "AV", "BA", "BC", "BY", "CA", "CL", "CM", "CO", "CR", "CS", "CT", "DA", "DC", "DR", "EC", "EM", "ES", "FA", "FI", "FK", "FM", "FO", "FP", "FQ", "FR", "FV", "GC", "GD", "GE", "GF", "GFA", "GMO", "GG", "GU", "GW", "GX", "MF", "MGFA", "MGMO", "MH", "NE", "NN", "NR", "OS", "OV", "PA", "PB", "PC", "PO", "PP", "PQ", "PR", "PY", "PGFA", "PGMO", "RC", "SA", "SB", "SC", "SF", "SL", "SO", "SP", "SR", "SS", "ST", "STFA", "STMO", "TP", "UF", "UH", "UK", "UP", "XX"
    ]
    for idx, val in enumerate(abbrevs, 1):
        Abbrev[idx] = val
    # Allegation values (partial, expand as needed)
    Allegation[1][0] = "Neglectful Supervision"
    Allegation[1][1] = "neglectfully supervised"
    Allegation[1][2] = "For the allegation of NEGLECTFUL SUPERVISION, the preponderance of evidence standard is sufficient to state that"
    Allegation[1][3] = "For the allegation of NEGLECTFUL SUPERVISION, the preponderance of evidence standard is not sufficient to state that"
    Allegation[1][4] = "While there is sufficient information to determine that neglectful supervision occurred; the information  is not sufficient to state that the alleged perpetrator is responsible for it.  For the allegation of NEGLECTFUL SUPERVISION, the preponderance of evidence standard is not sufficient to state that"
    Allegation[1][5] = "The injuries/circumstances meet the definition of neglectful supervision as outlined in Texas Family Code §261.001 ..."
    # ... Add other Allegation rows as needed ...

def load_combo():
    # Placeholder for combo box logic (not needed in CLI)
    pass

def menu_transition(frm_focus=None, ctl_focus=None, menu_target=None):
    # Placeholder for menu transition logic (not needed in CLI)
    pass

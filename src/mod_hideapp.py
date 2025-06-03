# mod_hideapp.py
"""
Python equivalent of Mod_HideApp.bas
Contains window state constants and a stub for hiding/minimizing windows (not implemented for terminal mode).
"""

# Window state constants
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3

def six_hat_hide_window(nCmdShow):
    """
    Stub for window hiding/minimizing. Not implemented for terminal mode.
    """
    print(f"Window command: {nCmdShow} (no effect in terminal mode)")
    return True

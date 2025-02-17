#!/usr/bin/env python3
"""
AAVSO AVSpec FITS Validator

This script validates the headers of FITS files against AAVSO AVSpec compliance standards.
It checks for required and recommended keywords, ensuring they meet the expected format.
If errors are found, the script can attempt automatic corrections or prompt the user for manual input.

Features:
- Validates mandatory and recommended FITS keywords.
- Ensures correct data formats and value constraints.
- Supports automatic or interactive correction of errors.
- Works on Linux, macOS, and Windows.
- Does not modify the original file; saves corrected versions separately.

License: GPL3 License
Author: Mauro Barbieri
Date: 2025-02-17
Email: maurobarbieri.science AT gmail.com
"""

import sys
import os

# Cross-platform compatibility for Python execution
if sys.platform == "win32":
    python_cmd = "python"  # Windows
else:
    python_cmd = "python3"  # macOS/Linux

# Required packages
REQUIRED_PACKAGES = ["numpy", "astropy"]

def install_missing_packages():
    """Check for missing packages and install them automatically."""
    import importlib.util
    import subprocess

    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            print(f"Installing missing package: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install missing dependencies before proceeding
install_missing_packages()

import argparse
from astropy.io import fits
from astropy.time import Time
import numpy as np

##############################################################################
# Utility functions
##############################################################################

def normalize_datetime_format(time_str):
    """Normalize date-time formats to ISO-8601 format (YYYY-MM-DDTHH:MM:SS.sss)."""
    time_str = time_str.replace(" ", "T").replace("Z", "")
    try:
        t = Time(time_str, format="isot")
        return t.isot
    except:
        return None

def safe_float(x):
    """Attempt converting x to float, return None if invalid."""
    try:
        return float(x)
    except:
        return None

def safe_int(x):
    """Attempt converting x to int, return None if invalid."""
    try:
        return int(x)
    except:
        return None

def convert_to_bool_if_needed(value):
    """If a FITS boolean is stored as 'T' or 'F', convert to Python bool."""
    if isinstance(value, str) and value.strip().upper() in ('T', 'F'):
        return (value.strip().upper() == 'T')
    return value

def verbose_print(verbose, message):
    """Print message only if verbose is True."""
    if verbose:
        print(message)

def dry_run(parser):
    """Show a brief introduction and then the command-line usage from parser."""
    print("\nAVSpec FITS Validator\n")
    print("Validates FITS headers for required and recommended keywords, ensuring AVSpec compliance.")
    print("Displays a validation report of mandatory (and optionally recommended) keywords.")
    print("If --auto-correct is specified, it offers limited automated corrections or user input prompts.")
    print("A corrected file is saved with '_avspec_corrected.fits' appended. The original is never changed.\n")
    parser.print_help()
    print('\n')
    sys.exit(0)

##############################################################################
# Rules
##############################################################################

def validate_BITPIX(v, header, data):
    """
    Must be an integer multiple of 2 between -64 and 64, 
    and match the actual data's BITPIX if recognized.
    """
    if not (isinstance(v, int) and v % 2 == 0 and -64 <= v <= 64):
        return False

    # If we can infer BITPIX from data.dtype, check consistency
    inferred_bitpix = auto_correct_BITPIX(header, data)
    if inferred_bitpix is not None:
        return (v == inferred_bitpix)
    # If we cannot infer from data, at least it's in the correct numeric range
    return True

def validate_NAXIS1(v, header, data):
    """Must be equal to the length of the first axis in data."""
    if not isinstance(v, int):
        return False
    if data is None or data.ndim < 1:
        return False
    return (v == data.shape[-1])

def validate_DATE_END(v, header):
    """
    Must be in ISO format and strictly later than DATE-OBS.
    """
    if normalize_datetime_format(v) is None:
        return False
    try:
        t_end = Time(v).unix
        t_obs = Time(header.get("DATE-OBS", "1900-01-01T00:00:00")).unix
        return (t_end > t_obs)
    except:
        return False

def validate_EXPTIME(v, header):
    """
    Must be > 0 and <= (DATE-END - DATE-OBS).
    """
    if not isinstance(v, (int, float)) or v <= 0:
        return False
    try:
        t_obs = Time(header["DATE-OBS"]).unix
        t_end = Time(header["DATE-END"]).unix
        return ((t_end - t_obs) >= v)
    except:
        return False

def validate_AAV_ORD(v, header):
    """
    Required if AAV_SPKI == 'MULTI-ORDER'. Must be integer > 0.
    If AAV_SPKI != 'MULTI-ORDER', any value is considered valid or not required.
    """
    if header.get("AAV_SPKI") == "MULTI-ORDER":
        return (isinstance(v, int) and v > 0)
    return True

##############################################################################
# Auto-Correction Functions
##############################################################################

BITPIX_MAP = {
    np.dtype('uint8'): 8,
    np.dtype('int16'): 16,
    np.dtype('int32'): 32,
    np.dtype('int64'): 64,
    np.dtype('float32'): -32,
    np.dtype('float64'): -64
}

def auto_correct_SIMPLE(_header, _data):
    return True

def auto_correct_BITPIX(_header, data):
    """If we recognize the data dtype, return the correct BITPIX; otherwise None."""
    if data is None or data.dtype not in BITPIX_MAP:
        return None
    return BITPIX_MAP[data.dtype]

def auto_correct_NAXIS(_header, _data):
    return 1

def auto_correct_NAXIS1(_header, data):
    """Return data.shape[-1] if data is 1D or more."""
    if data is not None and data.ndim >= 1:
        return data.shape[-1]
    return None

def auto_correct_CRPIX1(header, data):
    return None

def auto_correct_AAV_SPKI(_header, _data):
    """
    Default to SINGLE-ORDER if unknown.
    """
    return "SINGLE-ORDER"

def auto_correct_AAV_ORD(header, data):
    """
    Only relevant if AAV_SPKI == 'MULTI-ORDER'. We cannot guess the actual order number => None.
    """
    if header.get("AAV_SPKI") == "MULTI-ORDER":
        return None
    return None

def auto_correct_DATE_OBS(_header, _data):
    """We cannot safely guess a date-time => None => must ask user if auto-correct is done interactively."""
    return None

def auto_correct_DATE_END(_header, _data):
    """Cannot safely guess => None => must ask user."""
    return None

def auto_correct_EXPTIME(header, data):
    """If DATE-OBS and DATE-END exist, guess difference, else None."""
    try:
        t_obs = Time(header["DATE-OBS"]).unix
        t_end = Time(header["DATE-END"]).unix
        diff = t_end - t_obs
        if diff > 0:
            return diff
    except:
        pass
    return None

def auto_correct_EXPTIME2(header, data):
    return None

def auto_correct_JD(header, data):
    """Compute midpoint JD if DATE-OBS and DATE-END are valid."""
    try:
        t_obs = Time(header["DATE-OBS"])
        t_end = Time(header["DATE-END"])
        return (t_obs.jd + t_end.jd) / 2.0
    except:
        return None

##############################################################################
# MANDATORY and RECOMMENDED RULES
##############################################################################

MANDATORY_RULES = {
    "SIMPLE": {
        "validate_fn": lambda v, h: (v is True),
        "error_msg": "Should be set to T (True).",
        "auto_correct_fn": auto_correct_SIMPLE
    },
    "BITPIX": {
        "validate_fn": validate_BITPIX,
        "error_msg": "Must be an even integer in [-64,64] matching the data type.",
        "auto_correct_fn": auto_correct_BITPIX
    },
    "NAXIS": {
        "validate_fn": lambda v, h: (v == 1),
        "error_msg": "Must be 1 for a 1D spectrum.",
        "auto_correct_fn": auto_correct_NAXIS
    },
    "NAXIS1": {
        "validate_fn": validate_NAXIS1,
        "error_msg": "Must match the length of the first data axis.",
        "auto_correct_fn": auto_correct_NAXIS1
    },
    "CTYPE1": {
        "validate_fn": lambda v, h: (v == "WAVE"),
        "error_msg": "Must be 'WAVE'.",
        "auto_correct_fn": lambda _h,_d: "WAVE"
    },
    "CUNIT1": {
        "validate_fn": lambda v, h: (v == "Angstrom"),
        "error_msg": "Must be 'Angstrom'.",
        "auto_correct_fn": lambda _h,_d: "Angstrom"
    },
    "CRPIX1": {
        "validate_fn": lambda v, h: (
            isinstance(v, int) 
            and v >= 1 
            and v <= h.get("NAXIS1", 0)
        ),
        "error_msg": "Must be an integer 1 ≤ CRPIX1 ≤ NAXIS1.",
        "auto_correct_fn": None
    },
    "CRVAL1": {
        "validate_fn": lambda v, h: (isinstance(v, float) and v >= 0),
        "error_msg": "Must be a non-negative real number.",
        "auto_correct_fn": None
    },
    "CDELT1": {
        "validate_fn": lambda v, h: (
            isinstance(v, (int,float))
            and v > 0
            and v < h.get("CRVAL1", float('inf'))
        ),
        "error_msg": "Must be a positive real number less than CRVAL1.",
        "auto_correct_fn": None
    },
    "OBJNAME": {
        "validate_fn": lambda v, h: (isinstance(v, str) and len(v.strip()) > 0),
        "error_msg": "Object name must be recognized by the Variable Star Index (VSX).",
        "auto_correct_fn": None
    },
    "DATE-OBS": {
        "validate_fn": lambda v, h: (normalize_datetime_format(v) is not None),
        "error_msg": "Must be in ISO-8601 format (YYYY-MM-DDTHH:MM:SS.sss).",
        "auto_correct_fn": auto_correct_DATE_OBS
    },
    "DATE-END": {
        "validate_fn": validate_DATE_END,
        "error_msg": "Must be in ISO-8601 format (YYYY-MM-DDTHH:MM:SS.sss) and later than DATE-OBS.",
        "auto_correct_fn": auto_correct_DATE_END
    },
    "EXPTIME": {
        "validate_fn": validate_EXPTIME,
        "error_msg": "Must be > 0 and <= (DATE-END - DATE-OBS).",
        "auto_correct_fn": auto_correct_EXPTIME
    },
    "OBSERVER": {
        "validate_fn": lambda v, h: (isinstance(v, str) and len(v.strip()) > 0),
        "error_msg": "Must match user login obscode (non-empty).",
        "auto_correct_fn": None
    },
    "AAV_SPKI": {
        "validate_fn": lambda v, h: (v in ["SINGLE-ORDER","MULTI-ORDER"]),
        "error_msg": "'MULTI-ORDER' for Echelle, 'SINGLE-ORDER' otherwise.",
        "auto_correct_fn": auto_correct_AAV_SPKI
    },
    "AAV_ORD": {
        "validate_fn": validate_AAV_ORD,
        "error_msg": "Required if AAV_SPKI=='MULTI-ORDER'. Must be integer > 0.",
        "auto_correct_fn": auto_correct_AAV_ORD
    }
}

RECOMMENDED_RULES = {
    "EXPTIME2": {
        "validate_fn": lambda v, h: (
            isinstance(v, (int,float))
            and v > 0
            and v < h.get("EXPTIME", float('inf'))
        ),
        "error_msg": "Must be a positive real number less than EXPTIME.",
        "auto_correct_fn": None
    },
    "JD": {
        "validate_fn": lambda v, h: (
            isinstance(v, (int,float)) 
            and abs(v - (Time(h["DATE-END"]).jd + Time(h["DATE-OBS"]).jd)/2) < 1e-5
        ),
        "error_msg": "Must be ((DATE-END + DATE-OBS)/2) in JD.",
        "auto_correct_fn": auto_correct_JD
    },
    "AAV_INST": {
        "validate_fn": lambda v, h: (
            isinstance(v, str) and len(v.strip()) > 0
        ),
        "error_msg": "Instrument name (non-empty).",
        "auto_correct_fn": None
    },
    "AAV_SITE": {
        "validate_fn": lambda v, h: (
            isinstance(v, str) and len(v.strip()) > 0
        ),
        "error_msg": "Site name (non-empty).",
        "auto_correct_fn": None
    },
    "AAV_ITRP": {
        "validate_fn": lambda v, h: (
            isinstance(v, str) and len(v.strip()) > 0
        ),
        "error_msg": "Resolving power (non-empty).",
        "auto_correct_fn": None
    },
    "VERSION": {
        "validate_fn": lambda v, h: (
            isinstance(v, str) and len(v.strip()) > 0
        ),
        "error_msg": "Software version (non-empty).",
        "auto_correct_fn": None
    }
}

##############################################################################
# Validation & Correction
##############################################################################

def check_keyword(key, rule, header, data):
    """
    Validate a single keyword without attempting correction.
    Returns (valid: bool, present: bool).
    """
    present = (key in header)
    if not present:
        return (False, present)

    verbose_print(verbose, f"Retrieve value and do basic conversions")
    val = convert_to_bool_if_needed(header[key])
    verbose_print(verbose, f"Attempt validation, some rules accept data")
    try:
        verbose_print(verbose, f"If rule expects data as well, pass it")
        return (rule["validate_fn"](val, header, data), True)
    except TypeError:
        verbose_print(verbose, f"If not, pass just (val, header)")
        return (rule["validate_fn"](val, header), True)
    except:
        return (False, True)

def collect_issues(header, data, check_recommended=False):
    """
    Returns two lists: mandatory_issues and recommended_issues,
    each a list of (key, is_present, error_msg).
    We do NOT correct anything here; just gather the problems.
    """
    mandatory_issues = []
    recommended_issues = []

    verbose_print(verbose, f"Check mandatory")
    for key, rule in MANDATORY_RULES.items():
        is_valid, is_present = check_keyword(key, rule, header, data)
        if not is_valid:
            mandatory_issues.append((key, is_present, rule["error_msg"]))

    verbose_print(verbose, f"Check recommended if requested")
    if check_recommended:
        for key, rule in RECOMMENDED_RULES.items():
            is_valid, is_present = check_keyword(key, rule, header, data)
            if not is_valid:
                verbose_print(verbose, f"This covers either missing or present+invalid")
                recommended_issues.append((key, is_present, rule["error_msg"]))

    return mandatory_issues, recommended_issues

def user_prompt_fix(key, rule, header, data, is_mandatory):
    """
    Prompt the user for a valid value until they provide one or skip if not mandatory.
    Return True if eventually fixed, False otherwise.
    """
    error_msg = rule["error_msg"]
    while True:
        print(f"\n[{key}] {error_msg}")
        user_val = input(f"Enter a value for '{key}' (or press Enter to skip): ").strip()
        if user_val == "":
            if is_mandatory:
                print("This keyword is mandatory. You must provide a valid value.\n")
                continue
            else:
                verbose_print(verbose, f"For recommended, skipping is allowed")
                return False

        verbose_print(verbose, f"Attempt type casting based on known patterns")
        if "float" in error_msg.lower() or key in ["CRVAL1","CDELT1","EXPTIME","EXPTIME2","JD"]:
            typed_val = safe_float(user_val)
        elif "integer" in error_msg.lower() or key in ["NAXIS","NAXIS1","CRPIX1","BITPIX","AAV_ORD"]:
            typed_val = safe_int(user_val)
        elif key == "SIMPLE":
            typed_val = (user_val.upper() == "T")
        else:
            typed_val = user_val  # default as string

        verbose_print(verbose, f"Re-check validity")
        try:
            valid = False
            verbose_print(verbose, f"Attempt calling validation with data")
            validate_fn = rule["validate_fn"]
            verbose_print(verbose, f"Some rules expect (val, header, data)")
            try:
                valid = validate_fn(typed_val, header, data)
            except TypeError:
                valid = validate_fn(typed_val, header)
            
            if valid:
                header[key] = typed_val
                return True
        except:
            pass

        print(f"Invalid input for '{key}'. Please try again.\n")

def auto_correct_if_possible(key, rule, header, data):
    """
    Attempt to auto-correct using the rule's auto_correct_fn if any, returning True if corrected.
    Returns False otherwise (meaning we either have no auto_correct_fn or it returned None or invalid).
    """
    fn = rule["auto_correct_fn"]
    if fn is None:
        return False
    candidate = fn(header, data)
    if candidate is None:
        return False
    verbose_print(verbose, f"Validate candidate")
    try:
        validate_fn = rule["validate_fn"]
        verbose_print(verbose, f"Some accept data:")
        valid = False
        try:
            valid = validate_fn(candidate, header, data)
        except TypeError:
            valid = validate_fn(candidate, header)
        if valid:
            header[key] = candidate
            return True
    except:
        pass
    return False

def correct_issues(header, data, mandatory_issues, recommended_issues, interactive=False):
    """
    If interactive=True, attempt user fix or auto-correct for each issue in turn.
    If interactive=False, do nothing except return the final sets of unfixed issues.
    Returns updated sets of mandatory_issues, recommended_issues that remain.
    """
    verbose_print(verbose, f"Convert lists of issues to lists of (key, is_present, error_msg).")
    verbose_print(verbose, f"We'll rebuild the final still_issues after attempts to fix them.")

    verbose_print(verbose, f"We re-check each item after attempt to fix, removing from the remaining_issues if successful.")

    updated_mandatory = []
    for (key, present, msg) in mandatory_issues:
        if not interactive:
            verbose_print(verbose, f"If not interactive, no correction -> remain an issue")
            updated_mandatory.append((key, present, msg))
            continue

        verbose_print(verbose, f"Attempt auto-correct if possible")
        if present:  
            verbose_print(verbose, f"If the keyword is present but invalid, try auto-correct")
            if auto_correct_if_possible(key, MANDATORY_RULES[key], header, data):
                # Re-validate
                if check_keyword(key, MANDATORY_RULES[key], header, data)[0]:
                    continue  # fixed
        else:
            verbose_print(verbose, f"Key is missing, try auto-correct")
            if auto_correct_if_possible(key, MANDATORY_RULES[key], header, data):
                verbose_print(verbose, f"If now valid, good")
                if check_keyword(key, MANDATORY_RULES[key], header, data)[0]:
                    continue

        verbose_print(verbose, f"If auto-correct not successful, or no auto fn => prompt user")
        fixed = user_prompt_fix(key, MANDATORY_RULES[key], header, data, is_mandatory=True)
        verbose_print(verbose, f"Re-check")
        if not fixed:
            updated_mandatory.append((key, present, msg))

    updated_recommended = []
    for (key, present, msg) in recommended_issues:
        if not interactive:
            verbose_print(verbose, f"No correction if not interactive")
            updated_recommended.append((key, present, msg))
            continue

        if key not in RECOMMENDED_RULES:  # should not happen
            updated_recommended.append((key, present, msg))
            continue

        verbose_print(verbose, f"Attempt auto-correct if present/invalid or missing")
        if auto_correct_if_possible(key, RECOMMENDED_RULES[key], header, data):
            verbose_print(verbose, f"Check if valid")
            if check_keyword(key, RECOMMENDED_RULES[key], header, data)[0]:
                continue

        verbose_print(verbose, f"Otherwise prompt user (not mandatory => can skip)")
        fixed = user_prompt_fix(key, RECOMMENDED_RULES[key], header, data, is_mandatory=False)
        if not fixed:
            updated_recommended.append((key, present, msg))

    return updated_mandatory, updated_recommended

##############################################################################
# Main Validation Flow
##############################################################################

def validate_fits_file(fits_file, check_recommended=False, auto_correct=False, verbose=False):
    """
    1) Perform validation on mandatory (and if requested, recommended) keywords.
    2) Print a validation report with all issues.
    3) If auto_correct=True, interactively fix them (auto or user prompt).
    4) If any changes occur, write a new FITS file '<original>_avspec_corrected.fits'.
    """
    verbose_print(verbose, f"Reading FITS file: {fits_file}")
    corrected_filename = fits_file.replace(".fits", "_avspec_corrected.fits")

    with fits.open(fits_file, mode='readonly') as hdul:
        header_orig = hdul[0].header
        data_orig = hdul[0].data

        verbose_print(verbose, f"Phase 1: gather issues without any correction")
        mandatory_issues, recommended_issues = collect_issues(header_orig, data_orig, check_recommended=check_recommended)

        verbose_print(verbose, f"Print the validation report first")
        print("\nValidation Report:")
        if not mandatory_issues:
            print("  All mandatory keywords are valid.")
        else:
            print("  Mandatory issues:")
            for key, present, msg in mandatory_issues:
                if not present:
                    print(f"    - {key} [MISSING]: {msg}")
                else:
                    print(f"    - {key} [INVALID]: {msg}")

        if check_recommended:
            if not recommended_issues:
                print("  All recommended keywords are valid or absent.")
            else:
                print("  Recommended issues:")
                for key, present, msg in recommended_issues:
                    if not present:
                        print(f"    - {key} [MISSING]: {msg}")
                    else:
                        print(f"    - {key} [INVALID]: {msg}")

        verbose_print(verbose, f"If auto_correct == False => we stop here")
        if not auto_correct:
            return  # No corrections or new file creation

        verbose_print(verbose, f"If auto_correct == True => attempt to fix issues interactively")
        print("\nAttempting to fix issues (auto or manual input)...")

        verbose_print(verbose, f"Make an editable copy")
        header_new = header_orig.copy()
        data_new = data_orig.copy() if data_orig is not None else None

        verbose_print(verbose, f"Attempt corrections")
        updated_mandatory, updated_recommended = correct_issues(header_new, data_new, mandatory_issues, recommended_issues, interactive=True)

        verbose_print(verbose, f"Summarize which issues remain after correction")
        if updated_mandatory:
            print("\nThese mandatory issues remain UNFIXED:")
            for key, present, msg in updated_mandatory:
                if not present:
                    print(f"    - {key} [MISSING]: {msg}")
                else:
                    print(f"    - {key} [INVALID]: {msg}")
            print("Cannot proceed to full compliance with mandatory issues unresolved.")
        else:
            print("\nAll mandatory keywords appear valid or corrected now.")

        if check_recommended and updated_recommended:
            print("\nSome recommended issues remain unfixed:")
            for key, present, msg in updated_recommended:
                if not present:
                    print(f"    - {key} [MISSING]: {msg}")
                else:
                    print(f"    - {key} [INVALID]: {msg}")
        elif check_recommended:
            print("\nAll recommended keywords appear valid or corrected.")

        verbose_print(verbose, f"Compare final header to the original")
        if header_new != header_orig:
            new_hdu = fits.PrimaryHDU(data=data_new, header=header_new)
            new_hdulist = fits.HDUList([new_hdu])
            new_hdulist.writeto(corrected_filename, overwrite=True)
            print(f"\nA corrected file has been saved as '{corrected_filename}'.")
        else:
            print("\nNo changes applied, so no corrected file created.")

def main():
    global verbose
    verbose = False
    parser = argparse.ArgumentParser(
        description="FITS file keyword validator for AVSpec standards. "
                    "Checks mandatory (and optionally recommended) keywords. "
                    "Can optionally auto-correct or prompt for user fixes."
    )
    parser.add_argument("fits_file", nargs="?", help="Path to the FITS file to validate")
    parser.add_argument("--check-recommended", action="store_true", help="Also validate recommended keywords")
    parser.add_argument("--auto-correct", action="store_true", help="Enable interactive correction of errors")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if not args.fits_file:
        dry_run(parser)

    try:
        validate_fits_file(
            fits_file=args.fits_file,
            check_recommended=args.check_recommended,
            auto_correct=args.auto_correct,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\nError while validating '{args.fits_file}': {e}", file=sys.stderr)
    
    print('\n')

if __name__ == "__main__":
    main()

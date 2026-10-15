"""Root clean_data module proxy to scripts.clean_data."""
import sys
import scripts.clean_data
sys.modules[__name__] = scripts.clean_data

if __name__ == "__main__":
    cleaned, rep = scripts.clean_data.load_and_clean_faqs()
    print("\nCleaning report:")
    print(rep.summary())
    if rep.conflicts:
        print(f"Conflicts ({len(rep.conflicts)}):", rep.conflicts[:3])
    print(f"\nCleaned sample (first 3 rows):\n{cleaned.head(3)}")
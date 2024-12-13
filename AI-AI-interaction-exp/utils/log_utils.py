import io, warnings, sys
# Capture print, warning and store them in result. 
def silence_decorator(func, show_warnings=False, show_prints=False):
    def wrapper(*args, **kwargs):
        # Capture print statements
        original_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        # Capture warnings
        with warnings.catch_warnings(record=True) as captured_warnings:
            result = func(*args, **kwargs)

        # Print captured print statements
        sys.stdout = original_stdout
        if show_prints:
            print("=== Captured Output ===")
            print(captured_output.getvalue())
        if show_warnings:
            print("=== Captured Warnings ===")
            for warning in captured_warnings:
                print(warning.message)

        return result

    return wrapper
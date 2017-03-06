import sys
"""
python remove_empty_lines.py input_filename output_filename
"""

with open(sys.argv[1], 'r') as f_in:
    contents = f_in.readlines()
    with open(sys.argv[2], 'w') as f_out:
        for line in contents:
            if len(line) > 3:
                f_out.write(line)
    f_out.close()
f_in.close()